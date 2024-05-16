"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
from datasets import get_dataset
from models import get_model
from configs import *
from utils.file_utils import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

torch.backends.cudnn.enabled = True # Good
try:
    torch.set_float32_matmul_precision('medium') # medium is better
    print('set_float32_matmul_precision medium success')
except:
    print('set_float32_matmul_precision failed')

# same_seeds()
config_name = sys.argv[1]
args = eval(config_name).get_args()

model = get_model(args, args['model']['name'], args['model']['kwargs'])
try:
    model.init_output(args)
except:
    model.init_args(args)
if 'kd' in args['model']['kwargs'].keys() and  args['model']['kwargs']['kd']:
    model.reinit_kd_model()
if 'miner_model' in args['model']['kwargs'].keys() and  args['model']['kwargs']['miner_model']:
    model.reinit_miner_model()

def load_checkpoint(model_path):
    weights = torch.load(model_path, map_location=lambda storage, loc: storage)
    epoch = None
    if 'epoch' in weights:
        epoch = weights.pop('epoch')
    if 'state_dict' in weights:
        state_dict = (weights['state_dict'])
    else:
        state_dict = weights
    return epoch, state_dict


if "resume_path" in args.keys() and "re_train" in args.keys():
    ckpt = torch.load(args["resume_path"], map_location=lambda storage, loc: storage)['state_dict']
    if 'remove_key_words' in args.keys():
        ckpt = remove_key_word(ckpt, args['remove_key_words'])
    try:
        p = model.load_state_dict(ckpt)
        print(p)
    except:
        print('Load weights with strict False')
        # new_state_dict = remove_key_word(ckpt, ['model._fc.'])
        p = model.load_state_dict(ckpt, strict=False)
        print(p)
        # model.load_state_dict(ckpt, strict=False)
    print('Load weights successfully for re-train, %s' % args["resume_path"])

checkpoint_callback1 = ModelCheckpoint(
    monitor='acc5',
    dirpath=args['save_dir'],
    filename='best-{acc5:.4f}-{epoch:03d}' + '-max{:03d}'.format(args['n_epochs']),
    save_top_k=1,
    mode='max')
checkpoint_callback2 = ModelCheckpoint(
    monitor='val_loss',
    dirpath=args['save_dir'],
    filename='best-{val_loss:.8f}-{epoch:03d}' + '-max{:03d}'.format(args['n_epochs']),
    save_top_k=1,
    mode='min')
checkpoint_callback3 = ModelCheckpoint(
    monitor='acc1',
    dirpath=args['save_dir'],
    filename='best-{acc1:.4f}-{epoch:03d}' + '-max{:03d}'.format(args['n_epochs']),
    save_top_k=1,
    mode='max')
checkpoint_callback4 = ModelCheckpoint(
    monitor='epoch',
    dirpath=args['save_dir'],
    filename='{epoch:03d}' + '-max{:03d}'.format(args['n_epochs']),
    save_top_k=1,
    mode='max')
if 'train1' in args.keys() and  args['train1']:
    # train dataloader
    train_dataset = get_dataset(args['train_dataset']['name'], args['train_dataset']['kwargs'])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args['train_dataset']['batch_size'], shuffle=True, drop_last=True,
        num_workers=args['train_dataset']['workers'], pin_memory=False)

    # val dataloader
    val_dataset = get_dataset(args['val_dataset']['name'], args['val_dataset']['kwargs'])
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args['val_dataset']['batch_size'], shuffle=False, drop_last=False,
        num_workers=args['val_dataset']['workers'],
        pin_memory=args['val_dataset']['pin_memory'] if "pin_memory" in args['val_dataset'].keys() else False)

    trainer = pl.Trainer(max_epochs=args['n_epochs'],
                     val_check_interval=1.0 if 'val_check_interval' not in args.keys() else args['val_check_interval'],
                     check_val_every_n_epoch=1 if 'check_val_every_n_epoch' not in args.keys() else args['check_val_every_n_epoch'],
                     # strategy="ddp" if 'strategy' not in args.keys() else args['strategy'],
                     strategy="ddp" if 'strategy' not in args.keys() else args['strategy'],
                     devices=2 if 'gpus' not in args.keys() else args['gpus'],
                     accelerator='auto',
                     default_root_dir=args['save_dir'],
                     callbacks=[checkpoint_callback1, checkpoint_callback2, checkpoint_callback3, checkpoint_callback4],
                     precision=32 if 'precision' not in args.keys() else args['precision'],
                     num_sanity_val_steps=0,
                     sync_batchnorm=True,  # set to True when bs is too small on one card
                     gradient_clip_val=1.0 if 'gradient_clip_val' not in args.keys() else args['gradient_clip_val'],
                     gradient_clip_algorithm="norm")
    trainer.fit(model, train_loader, val_loader)

if 'train2' in args.keys() and args['train2']:
    # the second trainer
    if len(checkpoint_callback3.best_model_path) > 0:
        args["resume_path"] = checkpoint_callback3.best_model_path
    if 'fix_bn' in args: # for plate seg
        args['fix_bn'] = True
    args["lr"] = args["lr"] * 0.1
    # args["n_epochs"] = 40
    if 'weak_aug' in args['train_dataset']['kwargs'].keys():
        args['train_dataset']['kwargs']['weak_aug'] = True
    print(f'\n###The second training with lr {args["lr"]}, and weak aug')

    # reinit train dataloader
    train_dataset = get_dataset(args['train_dataset']['name'], args['train_dataset']['kwargs'])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args['train_dataset']['batch_size'], shuffle=True, drop_last=True,
        num_workers=args['train_dataset']['workers'], pin_memory=False)
    # val dataloader
    val_dataset = get_dataset(args['val_dataset']['name'], args['val_dataset']['kwargs'])
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args['val_dataset']['batch_size'], shuffle=False, drop_last=False,
        num_workers=args['val_dataset']['workers'],
        pin_memory=args['val_dataset']['pin_memory'] if "pin_memory" in args['val_dataset'].keys() else False)

    model = get_model(args, args['model']['name'], args['model']['kwargs'])
    try:
        model.init_output(args)
    except:
        model.init_args(args)

    if 'kd' in args['model']['kwargs'].keys() and  args['model']['kwargs']['kd']:
        model.reinit_kd_model()
    if 'miner_model' in args['model']['kwargs'].keys() and  args['model']['kwargs']['miner_model']:
        model.reinit_miner_model()

    if "resume_path" in args.keys() and "re_train" in args.keys():
        ckpt = torch.load(args["resume_path"], map_location=lambda storage, loc: storage)['state_dict']
        try:
            p = model.load_state_dict(ckpt)
            print(p)
        except:
            if 'remove_key_words' in args.keys():
                ckpt = remove_key_word(ckpt, args['remove_key_words'])
            print('Load weights with strict False')
            p = model.load_state_dict(ckpt, strict=False)
            print(p)
        print('Load weights successfully for re-train, %s' % args["resume_path"])
    checkpoint_callback1 = ModelCheckpoint(
        monitor='acc5',
        dirpath=args['save_dir'],
        filename='best-{acc5:.4f}-{epoch:03d}' + '-max{:03d}'.format(args['n_epochs']),
        save_top_k=1,
        mode='max')
    checkpoint_callback2 = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args['save_dir'],
        filename='best-{val_loss:.8f}-{epoch:03d}' + '-max{:03d}'.format(args['n_epochs']),
        save_top_k=1,
        mode='min')
    checkpoint_callback3 = ModelCheckpoint(
        monitor='acc1',
        dirpath=args['save_dir'],
        filename='best-{acc1:.4f}-{epoch:03d}' + '-max{:03d}'.format(args['n_epochs']),
        save_top_k=1,
        mode='max')
    checkpoint_callback4 = ModelCheckpoint(
        monitor='epoch',
        dirpath=args['save_dir'],
        filename='{epoch:03d}' + '-max{:03d}'.format(args['n_epochs']),
        save_top_k=1,
        mode='max')
    trainer = pl.Trainer(max_epochs=args['n_epochs'],
                     val_check_interval=1.0 if 'val_check_interval' not in args.keys() else args['val_check_interval'],
                     check_val_every_n_epoch=1 if 'check_val_every_n_epoch' not in args.keys() else args['check_val_every_n_epoch'],
                     # strategy="ddp" if 'strategy' not in args.keys() else args['strategy'],
                     strategy="ddp" if 'strategy' not in args.keys() else args['strategy'],
                     devices=2 if 'gpus' not in args.keys() else args['gpus'],
                     accelerator='auto',
                     default_root_dir=args['save_dir'],
                     callbacks=[checkpoint_callback1, checkpoint_callback2, checkpoint_callback3, checkpoint_callback4],
                     precision=32 if 'precision' not in args.keys() else args['precision'],
                     num_sanity_val_steps=-1,
                     sync_batchnorm=True,  # set to True when bs is too small on one card
                     gradient_clip_val=1.0 if 'gradient_clip_val' not in args.keys() else args['gradient_clip_val'],
                     gradient_clip_algorithm="norm")

    trainer.fit(model, train_loader, val_loader)
    print(f'###The second training finished!!! The best model is {checkpoint_callback3.best_model_path}')

if 'final_test' in args.keys() and args['final_test']:
    if args['train1'] or args['train2']:
        ckpt_path = checkpoint_callback3.best_model_path
    else:
        ckpt_path = args["resume_path"]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from utils.torch_utils import same_seeds
    same_seeds()
    model = get_model(args, args['model']['name'], args['model']['kwargs'])
    try:
        model.init_output(args)
    except:
        model.init_args(args)
    trainer = pl.Trainer(max_epochs=args['n_epochs'],
                             strategy="ddp", gpus=1 if 'gpus' not in args.keys() else args['gpus'],
                             default_root_dir=args['save_dir'],
                             num_sanity_val_steps=0,
                             precision=32 if 'precision' not in args.keys() else args['precision'], )
    print('Testing Best Model:', ckpt_path)
    # val dataloader
    val_dataset = get_dataset(args['val_dataset']['name'], args['val_dataset']['kwargs'])
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args['val_dataset']['batch_size'], shuffle=False, drop_last=False,
        num_workers=args['val_dataset']['workers'],
        pin_memory=args['val_dataset']['pin_memory'] if "pin_memory" in args['val_dataset'].keys() else False)
    try:
        trainer.test(model, val_loader, ckpt_path, verbose=False)
    except:
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)['state_dict']
        try:
            p = model.load_state_dict(ckpt)
            print(p)
        except:
            if 'remove_key_words' in args.keys():
                ckpt = remove_key_word(ckpt, args['remove_key_words'])
            print('Load weights with strict False')
            p = model.load_state_dict(ckpt, strict=False)
            print(p)
        print('Load weights successfully for re-train, %s' % args["resume_path"])
        trainer.test(model, val_loader, verbose=False)
