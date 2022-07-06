from image_manipulation import imageManipulation

def main():
    im = imageManipulation('/Users/alexandrakim/Desktop/BUGS2022/example_movie.tif')
    im.standardDeviation()
    im.writeFile('output_image.tif')

if __name__ == "__main__":
    main()
