from fastbook import *

def main():
    urls = search_images_ddg('bird images', max_images=1)
    len(urls), urls[0]   
    dest = Path('bird.jpg')
    if not dest.exists(): 
        print(dest)
        download_url(urls[0], dest, show_progress=True)
    
    im = Image.open(dest)
    im.to_thumb(256, 256)
    
    searches = 'forest', 'bird'
    path = Path('bird_or_not')
    print(path)
    if not path.exists():
        for o in searches:
            dest = (path/o)
            print(dest)
            dest.mkdir(exist_ok=True, parents=True)
            results = search_images_ddg(f'{o} photo')
            download_images(dest, urls=results[:115])
            resize_images(dest, max_size=400, dest=dest)
    
    print(len(get_image_files(path)))
    failed = verify_images(get_image_files(path))
    print(failed.sum())
    failed.map(Path.unlink)
    
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=60),
        get_y=parent_label,
        item_tfms=[Resize(100, method='squish')]  # Increase image size to 224x224
    ).dataloaders(path, batch_size=32)  # Add explicit batch size

    print(f"Training batches: {len(dls.train)}")
    print(f"Validation batches: {len(dls.valid)}")
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(3, freeze_epochs=2)  # Train for more epochs
    
    # For prediction, make sure to process the image the same way as training
    img = PILImage.create('forest.jpg')
    is_bird, _, probs = learn.predict(img)
    print(f"This is a: {is_bird}")
    print(f"Probability: {probs[0]:.4f}")

if __name__ == '__main__':
    main()