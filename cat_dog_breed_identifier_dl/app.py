from fastai.vision.all import *
import gradio as gr
import timm

# Cell
learn = load_learner('model_catdog_breed_class.pkl')

# Cell
categories = learn.dls.vocab

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

# Cell
inputs = gr.Image(type="pil")
label = gr.Label()
examples = ['basset.jpg']

# Cell
intf = gr.Interface(fn=classify_image, inputs=inputs, outputs=label, examples=examples)
intf.launch()