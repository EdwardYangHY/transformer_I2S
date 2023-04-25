name_dict={
    'apple': ['apple','apples'],
    'banana': ['banana','bananas'],
    'carrot': ['carrot','carrots'],
    'grape': ['grape','grapes'],
    'cucumber': ['cucumber','cucumbers'],
    'egg': ['egg','eggs'],
    'eggplant': ['eggplant','eggplants'],
    'greenpepper': ['pepper','peppers','green pepper','green peppers'],
    'pea': ['pea','peas','green pea','green peas'],
    'kiwi': ['kiwi','kiwi fruit','kiwi fruits'],
    'lemon': ['lemon','lemons'],
    'onion': ['onion','onions'],
    'orange': ['orange','oranges'],
    'potatoes': ['potato','potatoes'],
    'bread': ['bread', 'sliced bread'],
    'avocado': ['avocado','avocados'],
    'strawberry': ['strawberry','strawberries'],
    'sweetpotato': ['sweet','sweet potato','sweet potatoes'],
    'tomato': ['tomato','tomatoes'],
    'turnip': ['radish','radishes','white radish','white radishes']
    #'orange02': '/orange02'
}
color_dict = {
    'wh': 'white',
    'br': 'brown',
    'bl': 'blue'
}
number_dict = {
    1: 'one',
    2: 'two',
    3: 'three'
}

def get_image_info(image_name):
    info=image_name
    name=info.split("_")[0]
    if name=="orange":
        color=info.split("_")[1]
        number=info.split("_")[2]
    else:
        color=info.split("_")[1][0:-1]
        number=info.split("_")[1][-1]
    return name, color, number

def judge_ans(transcription, image_name):
    ans = transcription.split(" ")
    # print(ans)
    right_name = False
    right_color = False
    right_number = False
    right_ans = False

    name, color, number = get_image_info(image_name)

    # 分开两个词 怎么办
    for an in ans:
        if an in name_dict[name]:
            # print("Right name.")
            right_name = True
        if an == color_dict[color]:
            # print("Right color.")
            right_color = True
        if an in number_dict[int(number)]:
            # print("Right number.")
            right_number = True
    if right_name and right_color and right_number:
        right_ans = True
    return right_ans, right_name