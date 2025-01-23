
from fastbook import *

def main():
    ims = search_images_ddg('grizzly bear')
    print(len(ims))
    dest = 'images/grizzly.jpg'
    download_url(ims[0], dest)
    im = Image.open(dest)
    im.to_thumb(128,128)
    
    bear_types = 'grizzly','black','teddy'
    path = Path('bears')

    if not path.exists():
        path.mkdir()
        for o in bear_types:
            dest = (path/o)
            dest.mkdir(exist_ok=True)
            results = search_images_bing(key, f'{o} bear')
            download_images(dest, urls=results.attrgot('contentUrl'))
            
    
    fns = get_image_files(path)
    fns
    failed = verify_images(fns)
    failed
    failed.map(Path.unlink);

if __name__ == '__main__':
    main()