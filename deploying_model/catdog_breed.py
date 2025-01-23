from fastai.vision.all import *
from pathlib import Path
import gradio as gr 

def check_is_cat(x): 
    return x[0].isupper()

def classify_image(img):  # Simplified function signature
    learn = load_learner('model.pkl')
    categories = ('Dog','Cat','Dont Know')
    # Convert to fastai image format
    img = PILImage.create(img)
    # Get prediction
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

def main():
    # Test the model first
    im = PILImage.create('images/cat.jpg')
    learn = load_learner('model.pkl')
    Y = learn.predict(im)
    print(Y)
    
    # Create interface components
    inputs = gr.Image(type="pil")
    outputs = gr.Label()  # Changed variable name from label to outputs
    examples = [['dog.jpg'], ['cat.jpg'], ['dunno.jpg']]  # Wrap each example in a list
    
    # Create and launch interface with correct parameter names
    intf = gr.Interface(
        fn=classify_image,
        inputs=inputs,      # Changed from input to inputs
        outputs=outputs,    # Changed from output to outputs
        examples=examples
    )
    
    # Launch the interface
    intf.launch(inline=True)
    
    # Model parameters (optional)
    m = learn.model
    ps = list(m.parameters())
    print(ps)

if __name__ == '__main__':
    main()