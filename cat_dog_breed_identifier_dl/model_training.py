from fastai.vision.all import *
import timm


def main(): 
    path = untar_data(URLs.PETS)/'images'
    #path = Path('oxford-iiit-pet').joinpath('images')
    print(path)

    dls = ImageDataLoaders.from_name_func('.',
        get_image_files(path), valid_pct=0.2, seed=42,
        label_func=RegexLabeller(pat = r'^([^/]+)_\d+'),
        item_tfms=Resize(224))
    
    timm.list_models('convnext*')
    
    learn = vision_learner(dls, 'convnext_tiny_in22k', metrics=error_rate).to_fp16()
    learn.fine_tune(3)
    
    learn.export('model_catdog_breed_class.pkl')
    current_dir = Path('.')
    model_path = current_dir/'model_catdog_breed_class.pkl'
    if not model_path.exists():
        shutil.copy(str(path/'model_catdog_breed_class.pkl'), str(model_path))
    print(f"Model saved to: {model_path.absolute()}")
    
if __name__ == '__main__':
    main()