from fastai.vision.all import *
from pathlib import Path

def check_is_cat(x): 
    return x[0].isupper()
def main(): 
    #path = untar_data(URLs.PETS)/'images'
    path = Path('oxford-iiit-pet').joinpath('images')
    print(path)
    
    dls = ImageDataLoaders.from_name_func(
        path, get_image_files(path), valid_pct=0.2, seed=42,
        label_func=check_is_cat, item_tfms=Resize(224))

    learn = vision_learner(dls, resnet34, metrics=error_rate)
    learn.fine_tune(3)
    learn.export('model.pkl')
    current_dir = Path('.')
    model_path = current_dir/'model.pkl'
    if not model_path.exists():
        shutil.copy(str(path/'model.pkl'), str(model_path))
    print(f"Model saved to: {model_path.absolute()}")


if __name__ == '__main__':
    main()