import json, base64

def encode_world_info(world_info):
    world_info_json = json.dumps(world_info).encode('utf-8')
    world_info_base64 = base64.urlsafe_b64encode(world_info_json).decode('utf-8')
    return world_info_base64
res = encode_world_info({'127.0.0.1' : [6]})
print(res)