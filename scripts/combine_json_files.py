from utils.file_utils import *

# root2 = "/media/fast_data/unified_json/"
# save_dst = "/media/fast_data/mid_json/all.json"
#
# def process(item):
#     if not item['image'].startswith('/'):
#         item['image'] = "/media/fast_data/unified_json/" +item['image']
#     if not os.path.isfile(item['image']):
#         # print('Not found', item['image'])
#         return None
#     try:
#         Image.open(item['image'])
#     except IOError:
#         # print('Bad Image', item['image'])
#         return None
#     item['ref'] = json_file
#     return item
#
# res = []
# json_files = make_dataset(root2, suffix=".json")
# print(json_files)
# for json_file in json_files:
#     info = load_json(json_file)
#     print(json_file)
#     process(info[0])
#     ans = run_imap_multiprocessing(process, info, 96)
#     res += [el for el in ans if el is not None]
#     print('Count:', len(res))
# save_json(save_dst, res)


# split json files
src_file = "/media/fast_data/mid_json/all.json"
dst_file = "/media/fast_data/mid_json/%d.json"
split_count = 9
info = load_json(src_file)
gap = len(info) // split_count
for i in range(split_count):
    dst_file_ = dst_file % i
    print(i, dst_file_)
    save_json(dst_file_, info[i*gap:i*gap+gap])
'''
/media/fast_data/huggingface/hub/datasets--FreedomIntelligence--ALLaVA-4V/snapshots/a86c85a0076c7b0e9c6c08a55ef314461d45db65/ALLaVA-Instruct-VFLAN-4V.json
{'id': 'allava_vflan_inst_0', 'image': 'allava_vflan/images/images_191task_1k/VQA_counting_COCO_train2014_000000076081.jpg', 'conversations': [{'from': 'human', 'value': '<image>\nWhat could be the occasion for this gathering, based on the attire of the individuals and the setting?'}, {'from': 'gpt', 'value': 'The gathering appears to be a casual, social outing. The attire of the individuals is relaxed and not formal, suggesting that they are likely there to enjoy a meal together in a casual setting, possibly as friends or family. The presence of a young child and the broad smiles indicate a light-hearted, non-formal occasion.'}]}
/media/fast_data/huggingface/hub/datasets--FreedomIntelligence--ALLaVA-4V/snapshots/a86c85a0076c7b0e9c6c08a55ef314461d45db65/ALLaVA-Caption-VFLAN-4V.json
{'id': 'allava_vflan_cap_0', 'image': 'allava_vflan/images/images_191task_1k/VQA_counting_COCO_train2014_000000076081.jpg', 'conversations': [{'from': 'human', 'value': '<image>\nElaborate on the details of the image provided.'}, {'from': 'gpt', 'value': "The image captures a group of four people seated at a table in a well-lit restaurant with large windows that offer a view of a landscaped area with lush greenery. On the left, a woman with blonde hair smiles at the camera, holding a stemmed glass with a dark red beverage, possibly wine. Next to her, a man is seated looking towards a woman across the table, neither directly facing the camera. The man has short-cropped hair and a light beard, wearing a casual buttoned shirt. Across the table, there's a woman with glasses, reddish hair, and a bright pink shirt, engaging in conversation with the man. On the right, a young girl with blonde hair smiles broadly at the camera, wearing an orange sleeveless top. She's holding a menu. On the table, there are four glasses with varying levels of water, another glass with a red beverage, and a small plate with a strawberry on the rim, possibly belonging to a dessert or a cocktail. Personal items, such as a black handbag and a pair of sunglasses, are also on the table."}]}
/media/fast_data/huggingface/hub/datasets--FreedomIntelligence--ALLaVA-4V/snapshots/a86c85a0076c7b0e9c6c08a55ef314461d45db65/Evol-Instruct-GPT4-Turbo-143K.json
{'idx': 'heR0vZB', 'conversations': [{'from': 'human', 'value': 'As an online platform teacher named Aimee, you possess impeccable credentials which include a Bachelor of Science degree in Industrial and Labor Relations from Cornell University, expertise in the English language, and intermediate proficiency in both Chinese and Spanish. Additionally, your professional experience as a STEAM teacher at UN Women in Singapore has honed your skills in teaching children from the ages of 6-11 and working with students from all levels of education. Your exceptional teaching abilities in spoken English and pronunciation paired with your personal strengths of being informed, patient, and engaging make you an ideal teacher for students seeking to improve their English language skills. Can you provide a short, concise, and unique English self-introduction in bullet point form that would attract students to enroll in your course?'}, {'from': 'gpt', 'value': "Sure, here are some bullet points for your English self-introduction:\n\n- Hi there! My name is Aimee and I'm thrilled to be your online English teacher!\n- With a Bachelor's degree from Cornell University and experience teaching STEAM at UN Women in Singapore, I know how to help students learn effectively.\n- I'm passionate about language and have intermediate proficiency in both Chinese and Spanish, so I understand the challenges of learning a new language.\n- In my classes, I focus on improving spoken English and pronunciation, as well as building vocabulary and grammar skills.\n- But learning English doesn't have to be boring! I'm an engaging and patient teacher who loves to have fun and make learning enjoyable.\n- So, whether you're a beginner or an advanced student, I'm confident that I can help you improve your English skills and achieve your goals. Let's get started!"}]}
/media/fast_data/huggingface/hub/datasets--FreedomIntelligence--ALLaVA-4V/snapshots/a86c85a0076c7b0e9c6c08a55ef314461d45db65/ALLaVA-Instruct-LAION-4V.json
{'id': 'allava_laion_inst_0', 'conversations': [{'from': 'human', 'value': '<image>\nConsidering the items the figurine is holding, what scene or event from its source material might it be representing?'}, {'from': 'gpt', 'value': 'Based on the items the figurine is holding, which are a rifle and an iron, it likely represents a scene from the source material where the character is setting up traps to protect his home. These items are indicative of the resourceful and inventive methods the character uses in the movie "Home Alone" to create humorous but effective traps for intruders.'}], 'original_caption': 'Фигурка Funko POP! Vinyl: Home Alone: Kevin 21778', 'url': 'https://i3.stat01.com/2/490/104897720/075a3e/fun676-jpg.jpg'}
/media/fast_data/huggingface/hub/datasets--FreedomIntelligence--ALLaVA-4V/snapshots/a86c85a0076c7b0e9c6c08a55ef314461d45db65/ALLaVA-Caption-LAION-4V.json
{'id': 'allava_laion_cap_0', 'conversations': [{'from': 'human', 'value': '<image>\nPlease provide a vivid description of the image.'}, {'from': 'gpt', 'value': 'The image displays a figurine inside its packaging. The packaging is a box that is predominantly white with a clear plastic window allowing the figurine to be viewed. The box is branded with "Pop! Movies" at the top and "Home Alone" beneath it, indicating the movie theme of the collectible. The figure is labeled with the name "Kevin," which is printed in red font, and it\'s numbered 491 in its series. The figure inside resembles a young boy with simplistic features characteristic of the Funko Pop! style: large black eyes, a small nose, and no mouth. The hair is sculpted in a short, blonde style. The figurine is dressed in a red sweater with a quilted design and green pants. It\'s accessorized with two items, a rifle held in the left hand and an iron in the right hand. The iron and rifle are relatively plain with minimal details. The figure stands upright on its own, with brown shoes visible beneath the green pants. The box has red side panels and a green front bottom panel with additional branding and warnings typical of collectible packaging. There is a warning label indicating a choking hazard.'}], 'original_caption': 'Фигурка Funko POP! Vinyl: Home Alone: Kevin 21778', 'url': 'https://i3.stat01.com/2/490/104897720/075a3e/fun676-jpg.jpg'}

https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k
/home/xuzhenbo/.cache/huggingface/hub/datasets--WizardLM--WizardLM_evol_instruct_V2_196k/snapshots/29f88cf83c7e784cd942d26959d29192aa2cbdb1
https://huggingface.co/datasets/Vision-Flan/vision-flan_191-task_1k/tree/main

'''



"""
### Sample Demonstration
* ALLaVA-Caption-LAION-4V
```
{
  "id": "allava_laion_cap_1",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nPlease depict the image in words."
    },
    {
      "from": "gpt",
      "value": "The image displays a silver ring with a design that resembles a stylized leaf. The leaf design is spli
tinto two symmetrical halves, separated by a gap that runs along the ring's band. Each half of the leaf pattern is text
ued with detailed, vein-like striations that give it a realistic look. The edges of the leaf design are embellished wit
hsmall, clear gemstones that could possibly be diamonds, adding a sparkle to the ring. The band of the ring is wide, su
porting the large leaf motif, and it tapers slightly as it moves away from the leaf design. The silver metal has a high
 olish, reflecting light brightly and providing a contrast to the textured leaf surface. The craftsmanship suggests a m
i of organic inspiration with a touch of elegance due to the gemstone accents."
    }
  ],
  "original_caption": "190952cz Pandora Tropical Palm Leaf Ring",
  "url": "https://theartofpandora.com/wp-content/uploads/2016/10/190952CZ-1-600x600.jpg"
}
  ```

* ALLaVA-Caption-VFLAN-4V
```
{
  "id": "allava_vflan_cap_100000",
  "image": "allava_vflan/images/images_191task_1k/PACS+guitar_image_category_classification_830_n03272010_10795-5.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nHow would you describe the image to someone who cannot see it?"
    },
    {
      "from": "gpt",
      "value": "The image shows a simplistic line drawing of an electric guitar. The guitar is oriented diagonally from the top left corner to the bottom right corner of the image. It has a long, narrow neck with tuning pegs at the top, resembling a stylized \"T\" shape. The body of the guitar is irregular, with two cutaways on either side of the neck, giving it an asymmetric appearance. The body is also characterized by sharp angles rather than curves, which is not typical for most guitars. There are four horizontal lines across the body, likely representing strings. The drawing is black and white, with no shading or color, and appears to be hand-drawn with a single continuous line without much detail. The background is plain and white."
    }
  ]
}
```

* ALLaVA-Instruct-LAION-4V

```
{
  "id": "allava_laion_inst_1",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nHow does the inclusion of gemstones affect the overall aesthetic and possible value of the ring?"
    },
    {
      "from": "gpt",
      "value": "The inclusion of gemstones along the edges of the leaf design enhances the ring's aesthetic by adding a touch of brilliance and luxury. The small, clear gemstones catch the light, creating a sparkling effect that contrasts with the silver's polished surface and the leaf's textured details. This sparkling accent likely increases the ring's visual appeal, making it stand out more as an accessory. In terms of value, the presence of gemstones, which could be diamonds, would likely raise the ring's monetary worth. Gemstones, especially diamonds, are prized for their beauty and durability, and their addition to jewelry pieces often signifies a higher quality and class of the item, thereby increasing its market value."
    }
  ],
  "original_caption": "190952cz Pandora Tropical Palm Leaf Ring",
  "url": "https://theartofpandora.com/wp-content/uploads/2016/10/190952CZ-1-600x600.jpg"
}
```

* ALLaVA-Instruct-VFLAN-4V
```
{
  "id": "allava_vflan_inst_9",
  "image": "allava_vflan/images/images_191task_1k/VQA-E+visual_question_answering_574_COCO_train2014_000000119456.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nYou are provided with an image and a question related to the image. Answer the question based on the information given in the image.\n\nIs bigfoot in the picture?"
    },
    {
      "from": "gpt",
      "value": "The creature in the image is not Bigfoot, also known as Sasquatch. Bigfoot is typically described in folklore and cryptozoology as a large, bipedal humanoid creature with ape-like features, standing upwards of 6 to 9 feet tall. The animal present in the photograph is quadrupedal, possesses a body structure similar to that of a pig, and has features such as long hair and tusks, which are characteristic of a wild boar. Additionally, there is no evidence of bipedal creatures or any human-like figures that would suggest the presence of Bigfoot. The setting and visual clues do not align with the commonly reported sightings of Bigfoot, which often describe it as a solitary figure standing upright among trees. Therefore, based on the visual information available in the image, the conclusion is that Bigfoot is not depicted here."
    }
  ]
}
```

* Evol-Instruct-GPT4-Turbo-143K
```
{
  "idx": "heR0vZB",
  "conversations": [
    {
      "from": "human",
      "value": "As an online platform teacher named Aimee, you possess impeccable credentials which include a Bachelor of Science degree in Industrial and Labor Relations from Cornell University, expertise in the English language, and intermediate proficiency in both Chinese and Spanish. Additionally, your professional experience as a STEAM teacher at UN Women in Singapore has honed your skills in teaching children from the ages of 6-11 and working with students from all levels of education. Your exceptional teaching abilities in spoken English and pronunciation paired with your personal strengths of being informed, patient, and engaging make you an ideal teacher for students seeking to improve their English language skills. Can you provide a short, concise, and unique English self-introduction in bullet point form that would attract students to enroll in your course?"
    },
    {
      "from": "gpt",
      "value": "Sure, here are some bullet points for your English self-introduction:\n\n- Hi there! My name is Aimee and I'm thrilled to be your online English teacher!\n- With a Bachelor's degree from Cornell University and experience teaching STEAM at UN Women in Singapore, I know how to help students learn effectively.\n- I'm passionate about language and have intermediate proficiency in both Chinese and Spanish, so I understand the challenges of learning a new language.\n- In my classes, I focus on improving spoken English and pronunciation, as well as building vocabulary and grammar skills.\n- But learning English doesn't have to be boring! I'm an engaging and patient teacher who loves to have fun and make learning enjoyable.\n- So, whether you're a beginner or an advanced student, I'm confident that I can help you improve your English skills and achieve your goals. Let's get started!"
    }
  ]
}
```
"""