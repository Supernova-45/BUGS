from image_manipulation import imageManipulation, arrayManipulation

def main():
    im = imageManipulation('/Users/alexandrakim/Desktop/BUGS2022/example_movie.tif')
    print(im.getArr())

if __name__ == "__main__":
    main()
