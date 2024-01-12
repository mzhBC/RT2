# 导入所需库
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModel
import tensorflow_datasets as tfds
from datasets import Dataset, load_from_disk
import os

# 模型和分词器的下载与加载
# 注意：这里注释掉的代码行是为了在需要时下载模型
model_dir = snapshot_download('goldsj/blip2-opt-2.7b', cache_dir='/Data/RT2/blip2-opt-2.7b')

# 从本地路径加载分词器和模型
tokenizer_path = "/Data/RT2/blip2-opt-2.7b/goldsj/blip2-opt-2.7b/"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
model = AutoModel.from_pretrained(tokenizer_path, trust_remote_code=True)

# 打印分词器和模型以确认加载
print(tokenizer)
print(model)

# 准备处理TensorFlow数据集
folder_path = '/Data/RT2/data/'
builder = tfds.builder_from_directory("/Data/0.1.0/")
bs = builder.as_dataset(split='train[:30000]')  # 获取训练数据的前30000条

# 处理并保存数据集
i = 0
for episode in bs:
    i += 1
    folder_episode_path = f"{folder_path}/episode{i}"
    if not os.path.exists(folder_episode_path):
        os.makedirs(folder_episode_path)

    # 提取每一步的图像和动作数据
    images = [step['observation']['image'].numpy() for step in episode['steps']]
    actions = [step["action"] for step in episode['steps']]
    world_vectors = [action["world_vector"].numpy() for action in actions]
    gripper_closedness_actions = [action["gripper_closedness_action"].numpy() for action in actions]
    terminate_episodes = [action['terminate_episode'].numpy() for action in actions]
    rotation_deltas = [action['rotation_delta'].numpy() for action in actions]
    natural_language_instructions = ["Instruction:" + step['observation']["natural_language_instruction"].numpy().decode('utf-8') + " Action :" for step in episode['steps']]

    # 组合数据为字典
    data_dict = {
        "images": images,
        "world_vectors": world_vectors,
        "gripper_closedness_actions": gripper_closedness_actions,
        "terminate_episodes": terminate_episodes,
        "rotation_deltas": rotation_deltas,
        "natural_language_instructions": natural_language_instructions,
    }

    # 将数据保存到磁盘
    dataset_dict = Dataset.from_dict(data_dict)
    dataset_dict.save_to_disk(folder_episode_path)

# 测试：从磁盘加载最后一个数据集
loaded_dataset = load_from_disk(folder_episode_path)
print(loaded_dataset)
