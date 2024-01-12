# 导入所需的库
from datasets import Dataset, load_from_disk, concatenate_datasets
import numpy as np
from gym import spaces
import action_tokenizer
from collections import OrderedDict
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model

# 设置 logging 配置
logging.basicConfig(level=logging.INFO, 
                    filename='training.log', 
                    filemode='w', 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 定义动作空间
output_tensor_space = spaces.Dict(
    OrderedDict(
        [
            ("terminate_episode", spaces.Discrete(4)),
            ("world_vector", spaces.Box(low=-0.05, high=0.05, shape=(3,), dtype=np.float32)),
            ("rotation_delta", spaces.Box(low=-np.pi / 10, high=np.pi / 10, shape=(3,), dtype=np.float32)),
            ("gripper_closedness_action", spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)),
        ]
    )
)

# 初始化动作分词器
vocab_size = 256
action_tokenizer = action_tokenizer.RT1ActionTokenizer(output_tensor_space, vocab_size=vocab_size)

# 加载和合并数据集
all_datasets = []
for i in range(1, 2500):  # 加载 1 到 2500 个数据集
    dataset_path = f'/Data/RT2/data/episode{i}'
    dataset = load_from_disk(dataset_path)
    all_datasets.append(dataset)

combined_dataset = concatenate_datasets(all_datasets)

# 定义图像字幕数据集类
class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["images"]
        image_array = np.array(image, dtype=np.uint8)
        pil_image = Image.fromarray(image_array)

        encoding = self.processor(pil_image, padding="max_length", return_tensors="pt")
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["natural_language_instructions"]

        action_sample = {
            "world_vector": torch.tensor(np.array(item["world_vectors"])).float(),
            "rotation_delta": torch.tensor(np.array(item["rotation_deltas"])).float(),
            "gripper_closedness_action": torch.tensor(np.array(item["gripper_closedness_actions"])).float(),
            "terminate_episode": torch.tensor(np.array(item["terminate_episodes"]).argmax(-1)).long(),
        }
        action = action_tokenizer.tokenize(action_sample)
        encoding["action"] = action
        return encoding

# 数据加载器 collate 函数
def collate_fn(batch):
    processed_batch = {key: torch.stack([example[key] for example in batch]) if key != "text" else processor.tokenizer(
        [example["text"] for example in batch], padding=True, return_tensors="pt") for key in batch[0]}
    return processed_batch

# 加载处理器和模型
processor = AutoProcessor.from_pretrained("/Data/RT2/blip2-opt-2.7b/goldsj/blip2-opt-2.7b/")
model = Blip2ForConditionalGeneration.from_pretrained(
    "/Data/RT2/blip2-opt-2.7b/goldsj/blip2-opt-2.7b/", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)

# 配置 LoraConfig 并获取模型
lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", target_modules=["q_proj", "k_proj"])
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 准备训练数据
train_dataset = ImageCaptioningDataset(combined_dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=500, collate_fn=collate_fn)

# 设置优化器和训练设备
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.train()

# 训练模型
for epoch in range(200):
    logging.info("Epoch: %d", epoch)
    for idx, batch in enumerate(train_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        labels = batch.pop("action").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float16)

        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
        loss = outputs.loss
        logging.info("Loss: %s", loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 生成文本示例
image = Image.open("/Data/RT-1/episode1/image_11.png")
prompt = "Instruction: pick rxbar chocolate from bottom drawer and place on counter. Action :"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)
generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

# 保存训练后的模型
model.save_pretrained('/Data/RT2/modle')
