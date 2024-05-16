import os.path

from utils.file_utils import *
from config import *
import torch.quantization
from criterions.ce_loss import *
import pytorch_lightning as pl
from utils.cls_utils import accuracy
import torchmetrics


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_data(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class ClsLightning(pl.LightningModule):
    def __init__(self, num_classes, net_type='faster_vit', mixup=False, uncert=False, reg_cls=True, resolution=[224, 224]):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.args, self.criterion, self.milestones = None, None, None
        self.mixup = mixup
        self.reg_cls = reg_cls
        self.uncert = uncert
        self.net_type = net_type
        self.num_classes = num_classes
        self.test_loader = None
        self.test_loader2 = None

        if net_type == 'fbnet_c':
            # self.model = fbnet(net_type, pretrained=True)
            # self.model.head.conv = nn.Conv2d(1984, num_classes, 1)
            b = 1
        elif net_type == 'faster_vit3':
            from models.faster_vit_any_res import faster_vit_3_any_res
            # self.mid_feat_len = 1024
            self.model = faster_vit_3_any_res(pretrained=True)
            self.model.head = nn.Linear(self.model.num_features, num_classes)
        elif net_type == 'faster_vit1':
            from models.faster_vit_any_res import faster_vit_1_any_res
            # self.mid_feat_len = 640
            self.model = faster_vit_1_any_res(pretrained=True)
            self.model.head = nn.Linear(self.model.num_features, num_classes)
        elif net_type == 'faster_vit':
            from models.faster_vit_any_res import faster_vit_2_any_res
            # self.mid_feat_len = 768
            self.model = faster_vit_2_any_res(pretrained=True, resolution=resolution)
            self.model.head = nn.Linear(self.model.num_features, num_classes)

        from utils.cls_utils import AccuracyTopK
        self.valid_acc = torchmetrics.Accuracy()
        self.valid_top5_acc = AccuracyTopK(top_k=2)

        print(f'ClsLightning with {self.net_type}')


    def _init_weights(self, m):
        from timm.models.layers import trunc_normal_
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def init_args(self, args):
        self.args = args
        if 'SmoothCE' in args['loss_type']:
            self.criterion = SmoothCE()
        elif 'FocalOHEMLoss' in args['loss_type']:
            self.criterion = FocalOHEMLoss(num_classes=self.num_classes)
        elif 'CrossEntropy' in args['loss_type']:
            self.criterion = torch.nn.CrossEntropyLoss()
        elif 'loss_gls' in args['loss_type']:
            from criterions.ce_loss import loss_gls
            self.criterion = loss_gls
            # self.criterion = lambda y, gt: loss_gls(y, gt, hard_ratio=0.3)
        self.milestones = args['milestones'] if 'milestones' in args.keys() else None
        self.lr = args['lr']

    def sgd_optimizer(self, lr, momentum, weight_decay):
        params = []
        for key, value in self.named_parameters():
            if not value.requires_grad:
                continue
            apply_weight_decay = weight_decay
            apply_lr = lr
            if value.ndimension() < 2:  # TODO note this
                apply_weight_decay = 0
                # print('set weight decay=0 for {}'.format(key))
            if 'bias' in key:
                apply_lr = 2 * lr  # Just a Caffe-style common practice. Made no difference.
            params += [{'params': [value], 'lr': apply_lr, 'weight_decay': apply_weight_decay}]
        optimizer = torch.optim.SGD(params, lr, momentum=momentum)
        return optimizer

    def adamw_optimizer(self, lr, weight_decay):
        params = []
        parameters = self.model.named_parameters() if (self.kd or self.miner_model) else self.named_parameters()
        for key, value in parameters:
            if not value.requires_grad:
                continue
            apply_weight_decay = weight_decay
            apply_lr = lr
            if value.ndimension() < 2:  # TODO note this
                apply_weight_decay = 0
                # print('set weight decay=0 for {}'.format(key))
            for el in list(self.model.no_weight_decay()):
                if el in key:
                    apply_weight_decay = 0
            if 'head' in key or 'fc' in key:
                apply_lr = 10 * lr
            elif 'bias' in key:
                apply_lr = 2 * lr  # Just a Caffe-style common practice. Made no difference.
            params += [{'params': [value], 'lr': apply_lr, 'weight_decay': apply_weight_decay}]
        if 'strategy' in self.args.keys() and self.args['strategy'] == 'deepspeed_stage_2_offload':
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            optimizer = DeepSpeedCPUAdam(params, lr)
        else:
            optimizer = torch.optim.AdamW(params, lr)
        return optimizer

    def configure_optimizers(self):
        params = list(self.named_parameters())

        if self.args['opt_param'] == 'fix_bn':
            def is_bn(n):
                return 'bn' in n

            param_wo_bn = [p for n, p in params if not is_bn(n)]
            grouped_parameters = [
                {"params": param_wo_bn, 'lr': self.args['lr']},
            ]
            print('BN Fixed!')
        elif self.args['opt_param'] == 'fc_only':
            def is_fc(n):
                return 'fc' in n

            param_fc = [p for n, p in params if is_fc(n)]
            grouped_parameters = [
                {"params": param_fc, 'lr': self.args['lr']},
            ]
            print('fc only!')
        elif self.args['opt_param'] == 'two_lr':
            def is_fc(n):
                return 'classifier' in n

            grouped_parameters = [
                {"params": [p for n, p in params if is_fc(n)], 'lr': self.args['lr']},
                {"params": [p for n, p in params if not is_fc(n)], 'lr': self.args['lr'] * 0.1}
            ]
            print('two lr!')
        else:  # e2e
            grouped_parameters = self.parameters()

        if 'adam' in self.args['opt_type'].lower():
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.args['lr'])
            # optimizer = torch.optim.AdamW(self.model.head.parameters(), lr=self.args['lr'])
        elif self.args['opt_type'].lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(grouped_parameters, lr=self.args['lr'])
        else:
            optimizer = self.sgd_optimizer(self.args['lr'], momentum=0.9, weight_decay=1e-4)

        if self.milestones is None:
            def lambda_(epoch):
                return pow((1 - (epoch / self.args['n_epochs'])), 0.9)

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_, )
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.milestones, gamma=0.3)
        return [optimizer], [scheduler]

    def forward(self,x):
        return self.model(x)

    def training_step(self, sample, batch_nb):
        # REQUIRED
        ims = sample['image']
        target = sample['label']

        if random.random() < 0.3:
        # if random.random() < 1.0:
            output = self(ims)
            loss = self.criterion(output, target)
        else:
            # cutmix # generate mixed sample
            beta=1.0
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(ims.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(ims.size(), lam)
            ims[:, :, bbx1:bbx2, bby1:bby2] = ims[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (ims.size()[-1] * ims.size()[-2]))
            # compute output
            output = self(ims)
            loss = self.criterion(output, target_a) * lam + self.criterion(output, target_b) * (1. - lam)
        info = {'loss': loss, 'focal_loss': loss, 'flow_loss': loss.detach()}
        self.training_step_outputs.append(info)
        self.log("loss", loss, prog_bar=True, on_epoch=True)  # , on_epoch=True is needed to be cleared each epoch
        return info

    def training_step_end(self, batch_parts):
        return {'loss': batch_parts['loss'].mean(), 'focal_loss': batch_parts['focal_loss'].mean(), 'flow_loss': batch_parts['flow_loss'].mean()}

    def on_train_epoch_end(self):
        if self.global_rank == 0:
            train_loss, focal_loss, flow_loss = 0, 0, 0
            train_step_outputs = self.training_step_outputs
            for v in train_step_outputs:
                train_loss += v['loss'].item()
                focal_loss += v['focal_loss'].item()
                flow_loss += v['flow_loss'].item()
            train_loss = train_loss / len(train_step_outputs)
            focal_loss = focal_loss / len(train_step_outputs)
            flow_loss = flow_loss / len(train_step_outputs)
            print('')
            print('===> train loss: {:.8f}, ===> focal_loss: {:.8f}, ===> flow_loss: {:.8f}'.format(train_loss,
                                                                                                    focal_loss,
                                                                                                    flow_loss))
            print('Learning Rate {:.8f}'.format(self.trainer.optimizers[0].param_groups[0]['lr']))

    def validation_step(self, sample, batch_idx):
        with torch.no_grad():
            im_path = sample['im_path'][0]
            ims = sample['image']
            target = sample['label']
            output = self(ims)

            loss = self.criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 1))
            self.valid_acc(output.softmax(dim=-1), target)
            self.valid_top5_acc(output.softmax(dim=-1), target)
            info = {'val_loss': loss, 'focal_loss': loss.detach()}
            self.validation_step_outputs.append(info)
            return info

    def on_validation_epoch_end(self,):
        validation_step_outputs = self.validation_step_outputs
        if len(validation_step_outputs) < 3:  # sanity check
            return
        val_loss, focal_loss, acc1, acc5 = 0, 0, 0, 0
        for v in validation_step_outputs:
            val_loss += v['val_loss'].item()
            focal_loss += v['focal_loss'].item()
        val_loss = val_loss / len(validation_step_outputs)
        focal_loss = focal_loss / len(validation_step_outputs)

        acc_avg = self.valid_acc.compute()
        acc5_avg = self.valid_top5_acc.compute()
        print('####### acc1: {:.4f}, acc5: {:.4f}, val seed: {:.8f}, val loss: {:.8f}'.format(acc_avg, acc5_avg, focal_loss, val_loss))
        self.log('acc1', acc_avg)
        self.log('acc5', acc5_avg)
        self.log('val_loss', val_loss)
        self.log('epoch', self.current_epoch)

    def test_step(self, sample, batch_idx):
        with torch.no_grad():
            im_path = sample['im_path'][0]
            ims = sample['image']
            target = sample['label']
            output = self(ims)

            loss = self.criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 1))
            self.valid_acc(output.softmax(dim=-1), target)
            self.valid_top5_acc(output.softmax(dim=-1), target)
            info = {'val_loss': loss, 'focal_loss': loss.detach()}
            self.validation_step_outputs.append(info)
            return info

    def on_test_epoch_end(self, ):
        validation_step_outputs = self.validation_step_outputs
        if len(validation_step_outputs) < 3:  # sanity check
            return
        val_loss, focal_loss, acc1, acc5 = 0, 0, 0, 0
        for v in validation_step_outputs:
            val_loss += v['val_loss'].item()
            focal_loss += v['focal_loss'].item()
        val_loss = val_loss / len(validation_step_outputs)
        focal_loss = focal_loss / len(validation_step_outputs)

        acc_avg = self.valid_acc.compute()
        acc5_avg = self.valid_top5_acc.compute()
        print('####### acc1: {:.4f}, acc5: {:.4f}, val seed: {:.8f}, val loss: {:.8f}'.format(acc_avg, acc5_avg,
                                                                                              focal_loss, val_loss))


if __name__ == '__main__':

    class export_clsLightning(nn.Module):
        def __init__(self, num_classes, net_type='fbnet_c', resolution=[384, 384]):
            super().__init__()
            self.args, self.criterion, self.milestones = None, None, None
            self.net_type = net_type
            self.num_classes = num_classes

            self.test_loader = None
            self.test_loader2 = None

            if net_type == 'fbnet_c':
                # self.model = fbnet(net_type, pretrained=True)
                # self.model.head.conv = nn.Conv2d(1984, num_classes, 1)
                b = 1
            elif net_type == 'faster_vit3':
                from models.faster_vit_any_res import faster_vit_3_any_res
                # self.mid_feat_len = 1024
                self.model = faster_vit_3_any_res(pretrained=False)
                self.model.head = nn.Linear(self.model.num_features, num_classes)
            elif net_type == 'faster_vit1':
                from models.faster_vit_any_res import faster_vit_1_any_res
                # self.mid_feat_len = 640
                self.model = faster_vit_1_any_res(pretrained=False)
                self.model.head = nn.Linear(self.model.num_features, num_classes)
            elif net_type == 'faster_vit':
                from models.faster_vit_any_res import faster_vit_2_any_res
                # self.mid_feat_len = 768
                self.model = faster_vit_2_any_res(pretrained=False, resolution=resolution)
                self.model.head = nn.Linear(self.model.num_features, num_classes)
            else:
                b = 1
            self.softmax = nn.Softmax(dim=-1)
            # self.quant = torch.quantization.QuantStub()
            # self.dequant = torch.quantization.DeQuantStub()

        def forward(self, x):
            y = self.model(x)
            return y

    inp_size = (384, 384)
    # ckpt_path = '/home/xuzhenbo/MoE-LLaVA/cls_hasFood/cls_hasFood/best-acc1=0.9727-epoch=059-max060.ckpt'
    ckpt_path = '/home/xuzhenbo/MoE-LLaVA/cls_hasFood/cls_hasFood/epoch=059-max060-v2.ckpt'
    save_jit_path = '/home/xuzhenbo/MoE-LLaVA/cls_hasFood/cls_hasFood/has_food_fv'


    print('-----------export gpu jit------------------')
    model = export_clsLightning(net_type='faster_vit', num_classes=2)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)['state_dict']
    # ckpt = remove_module_in_dict(ckpt)
    device = torch.device("cpu")
    model.eval()
    p = model.load_state_dict(ckpt, strict=True)
    print(p)
    device = torch.device("cuda")
    dummy_input = torch.rand(1, 3, inp_size[0], inp_size[1]).to(device)
    model.to(device)
    with torch.no_grad():
        torchscript_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(torchscript_model, save_jit_path + '_gpu.pt')

    print('-----------load real image------------------')
    model = torch.jit.load(save_jit_path + '_gpu.pt').to(device)
    from PIL import Image
    import torchvision.transforms as transforms
    from datasets.common_cls_dataset import SquarePad
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    transform = transforms.Compose([
        SquarePad(),
        transforms.Resize(inp_size, interpolation=3),  # BICUBIC interpolation
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    for real_input_path in sorted(make_dataset('/media/fast_data/food_image_test')):
        real_input = transform(Image.open(real_input_path).convert('RGB')).unsqueeze(0).to(device)
        re = model(real_input)
        re = torch.softmax(re, dim=-1)
        ans = re[0,1]>re[0,0]
        print(real_input_path, ans)

