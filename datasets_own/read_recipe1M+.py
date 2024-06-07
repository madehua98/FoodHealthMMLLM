# layer2+.json
'''
    {
        "id":"0c3db200f6",
        "images":[
            {
                "id":"c2ef35a26a.jpg",
                "url":"https://images.food52.com/vu311nkvDssjMHQ5U1bJTUXl_vk=/753x502/b8b3ccaf-1acd-414d-acdf-b6e391bdcd9d--DSC_1402.NEF.jpg"
            }
        ]
    },
    {
        "id":"2a51c2c80a",
        "images":[]
    }
]
'''
#recipe1M+_nutritional.json
'''
   {
      "id": "fffa89c6e3",
      "Fat": "6.245625g",
      "Protein": "7.220000g",
      "Mass": "199.437500g"
   },
   {
      "id": "fffb3bbff2",
      "Fat": "6700.970680g",
      "Protein": "301.603320g",
      "Mass": "26290.141730g"
   },
   {
      "id": "fffd4b124b",
      "Fat": "988.452986g",
      "Protein": "443.054586g",
      "Mass": "4149.592000g"
   },
   {
      "id": "ffffcd4444",
      "Fat": "18.226667g",
      "Protein": "3.245625g",
      "Mass": "180.600000g"
   }
]
'''
# recipes_with_nutritional_info.json
'''
{
        "fsa_lights_per100g":{
            "fat":"orange",
            "salt":"red",
            "saturates":"green",
            "sugars":"green"
        },
        "id":"ffffcd4444",
        "ingredients":[
            {
                "text":"radishes, raw"
            },
            {
                "text":"oil, sesame, salad or cooking"
            },
            {
                "text":"seeds, sesame seeds, whole, dried"
            },
            {
                "text":"salt, table"
            },
            {
                "text":"soy sauce made from soy (tamari)"
            }
        ],
        "instructions":[
            {
                "text":"Julienne the daikon and squeeze out the excess moisture."
            },
            {
                "text":"Combine all of the ingredients."
            },
            {
                "text":"Serve on a dish."
            }
        ],
        "nutr_per_ingredient":[
            {
                "fat":0.15,
                "nrg":24.0,
                "pro":1.02,
                "sat":0.048,
                "sod":58.5,
                "sug":2.7900000000000005
            },
            {
                "fat":13.6,
                "nrg":120.0,
                "pro":0.0,
                "sat":1.931,
                "sod":0.0,
                "sug":0.0
            },
            {
                "fat":4.47,
                "nrg":51.5625,
                "pro":1.595625,
                "sat":0.626125,
                "sod":1.0,
                "sug":0.026875
            },
            {
                "fat":0.0,
                "nrg":0.0,
                "pro":0.0,
                "sat":0.0,
                "sod":775.0,
                "sug":0.0
            },
            {
                "fat":0.006666666666666666,
                "nrg":3.6666666666666665,
                "pro":0.6299999999999999,
                "sat":0.0006666666666666666,
                "sod":335.0,
                "sug":0.10333333333333333
            }
        ],
        "nutr_values_per100g":{
            "energy":110.31515319306018,
            "fat":10.092284976005907,
            "protein":1.7971345514950166,
            "salt":1.6189091915836102,
            "saturates":1.4428525286083427,
            "sugars":1.6169481358434852
        },
        "partition":"train",
        "quantity":[
            {
                "text":"150"
            },
            {
                "text":"1"
            },
            {
                "text":"1"
            },
            {
                "text":"1/3"
            },
            {
                "text":"1"
            }
        ],
        "title":"Extremely Easy and Quick - Namul Daikon Salad",
        "unit":[
            {
                "text":"g"
            },
            {
                "text":"tablespoon"
            },
            {
                "text":"tablespoon"
            },
            {
                "text":"teaspoon"
            },
            {
                "text":"teaspoon"
            }
        ],
        "url":"https://cookpad.com/us/recipes/153324-extremely-easy-and-quick-namul-daikon-salad",
        "weight_per_ingr":[
            150.0,
            13.6,
            9.0,
            2.0,
            6.0
        ]
    }
]
'''
# train_recipe_image.json
'''
    {
        "id": "95c861aaca",
        "Fat": "18.226667g",
        "Protein": "3.245625g",
        "Mass": "180.600000g"
    },
    {
        "id": "26e07d2c83",
        "Fat": "18.226667g",
        "Protein": "3.245625g",
        "Mass": "180.600000g"
    }
'''
# train_nutrition_QA.json
'''
   {
      "id": "508629",
      "image": "/mnt/data_llm/food_images/recipe1M+/9/5/c/8/95c861aaca.jpg",
      "conversations": [
         {
            "from": "human",
            "value": "<image>\nPlease provide nutritional information for this dish."
         },
         {
            "from": "gpt",
            "value": "Mass: 180.600000g, Fat: 18.226667g, Protein: 3.245625g"
         }
      ]
   },
   {
      "id": "508630",
      "image": "/mnt/data_llm/food_images/recipe1M+/2/6/e/0/26e07d2c83.jpg",
      "conversations": [
         {
            "from": "human",
            "value": "<image>\nPlease provide nutritional information for this dish."
         },
         {
            "from": "gpt",
            "value": "Mass: 180.600000g, Fat: 18.226667g, Protein: 3.245625g"
         }
      ]
   }
]
'''