import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk, Image
import authtoken

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

#create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Diffusion Project Thesis")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(height=40,width=512, text_font=("Arial",20), text_color="black", fg_color="white", placeholder_text="Input Your Text Here")
prompt.place(x=10,y=10)

lmain = ctk.CTkLabel(height=512, width=512,text_font=("Arial",20), text_color="white")
lmain.place(x=10, y=110)


modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token="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")
pipe.to(device)


def generate():
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]

    image.save('generatedImage.png')
    generatedImage = Image.open("generatedImage.png")
    img = ImageTk.PhotoImage(generatedImage)
    lmain.configure(image=img)
    lmain.image = img

trigger = ctk.CTkButton(heigh=40, width=120, text_font=("Arial",20), text_color="white", fg_color="blue", text="Generate", command=generate)
trigger.place(x=206,y=60)


app.mainloop()
