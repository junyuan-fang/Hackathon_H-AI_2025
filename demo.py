from videoxl.model.builder import load_pretrained_model
from videoxl.mm_utils import tokenizer_image_token, process_images,transform_input_id
from videoxl.constants import IMAGE_TOKEN_INDEX,TOKEN_PERFRAME 
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
# fix seed
torch.manual_seed(0)


model_path = "/root/code/Video-XL/Video-XL/Video_XL/VideoXL_weight_8"
video_path="/root/code/Video-XL/433.MOV"

max_frames_num =900 # you can change this to several thousands so long you GPU memory can handle it :)
gen_kwargs = {"do_sample": True, "temperature": 1, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 1024}
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="cuda:0")

model.config.beacon_ratio=[8]   # you can delete this line to realize random compression of {2,4,8} ratio


prompt = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "<image>\n"
    "You are reviewing a video that shows a repair technician servicing an industrial elevator machine.  \n"
    "For **each distinct operation** the technician performs, produce a **step-by-step maintenance log** with the following structured fields:\n"
    "1. **Timestamp (Start → End)** (e.g. At 00:00 - 00:10)  \n"
    "2. **Machine Section / Component** – specify the exact area touched (e.g. upper drive pulley, gear housing, hydraulic line).  \n"
    "3. **Action Performed** – describe precisely what is done (e.g. loosens retaining bolt, removes worn V-belt, applies lithium-based grease).  \n"
    "4. **Tool(s) Used** – name every tool or instrument.  \n"
    "5. **Measurements / Settings** – record any values adjusted or readings taken (torque, gap, voltage, pressure, etc.).  \n"
    "6. **Parts Replaced or Serviced** – list part numbers or descriptions; note new vs. refurbished.  \n"
    "7. **Safety or Verification Checks** – describe tests, visual inspections, or instrument checks confirming proper function.  \n"
    "8. **Result & Recommendations** – state whether the issue was resolved and suggest follow-up maintenance if needed.\n\n"
    "Format the output as a **chronological table** or numbered list so another technician can replicate the work.\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
vr = VideoReader(video_path, ctx=cpu(0))
total_frame_num = len(vr)
uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
frame_idx = uniform_sampled_frames.tolist()
frames = vr.get_batch(frame_idx).asnumpy()
video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)

beacon_skip_first = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[1].item()
num_tokens=TOKEN_PERFRAME *max_frames_num
beacon_skip_last = beacon_skip_first  + num_tokens

with torch.inference_mode():
    output_ids = model.generate(input_ids, images=[video_tensor],  modalities=["video"],beacon_skip_first=beacon_skip_first,beacon_skip_last=beacon_skip_last, **gen_kwargs)

if IMAGE_TOKEN_INDEX in input_ids:
    transform_input_ids=transform_input_id(input_ids,num_tokens,model.config.vocab_size-1)

output_ids=output_ids[:,transform_input_ids.shape[1]:]
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(outputs)














#prompt = "Describe in detail the technician's actions and visual surroundings in the video. Include tool usage, elevator component being worked on, and any safety procedures visible."
# prompt = (
#     "<|im_start|>system\n"
#     "You are a helpful assistant.<|im_end|>\n"
#     "<|im_start|>user\n"
#     "<image>\n"
#     "You are reviewing a 15-minute overhead video of a technician servicing an **industrial elevator drive unit**.\n\n"
#     "➤ For **every clearly visible, distinct operation**, create a maintenance log **row** with the exact columns below.\n"
#     "➤ **Do NOT guess**.  If any detail is not directly observable or measurable in the video, write **UNKNOWN**.\n"
#     "➤ After filling the 8 fields, add a 9th column **Confidence** = High, Medium or Low (how certain you are the observation is correct).\n\n"
#     "| Timestamp (mm:ss–mm:ss) | Component | Action | Tool(s) | Measurement / Setting | Part(s) Replaced / Serviced | Safety / Verification | Result & Recommendation | Confidence |\n"
#     "|---|---|---|---|---|---|---|---|---|\n"
#     "- *Timestamp*: start–end of the specific action (e.g. 00:12–00:27).  \n"
#     "- *Component*: exact area touched (e.g. upper drive pulley, control PCB).  \n"
#     "- *Action*: concise verb phrase strictly describing what is **seen** (e.g. removes worn V-belt, tightens M8 bolt).  \n"
#     "- *Tool(s)*: list only tools **visibly used** (e.g. torque wrench 45 N·m).  \n"
#     "- *Measurement / Setting*: numeric readings or adjustments captured on-screen (gap 0.25 mm, 220 VAC…).  \n"
#     "- *Part(s)*: part names or numbers; mark NEW / REFURBISHED / CLEANED where applicable.  \n"
#     "- *Safety / Verification*: tests or inspections actually performed.  \n"
#     "- *Result & Recommendation*: immediate outcome + any follow-up advice.  \n"
#     "- *Confidence*: High / Medium / Low.\n\n"
#     "⚠️ If a column is not observable, write **UNKNOWN** in that cell.\n"
#     "⚠️ Output **only** the Markdown table, no extra commentary.\n"
#     "<|im_end|>\n"
#     "<|im_start|>assistant\n"
# )
#video input
# prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nDoes this video contain any inserted advertisement? If yes, which is the content of the ad?<|im_end|>\n<|im_start|>assistant\n"
# prompt = (
#     "<|im_start|>system\n"
#     "You are a helpful assistant.<|im_end|>\n"
#     "<|im_start|>user\n"
#     "<image>\n"
#     "This video features a repair technician at work. Identify every operation the technician performs and generate a detailed maintenance log that includes:\n"
#     "• Timestamps for each step\n"
#     "• Actions taken and tools used\n"
#     "• Parts repaired or replaced\n"
#     "• Testing or verification procedures\n"
#     "• Final outcome and any recommendations\n"
#     "<|im_end|>\n"
#     "<|im_start|>assistant\n"
# )