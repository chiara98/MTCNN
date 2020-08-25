from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import os

count = 0


def draw_boxes(results, directory, file):
    data = pyplot.imread(directory)
    pyplot.imshow(data)
    ax = pyplot.gca()
    for result in results:
        x, y, width, height = result['box']
        rect = Rectangle((x,y), width, height, fill=False, color='red')
        ax.add_patch(rect)
    pyplot.savefig('deteceted_face'+file)
    pyplot.show()
    pyplot.close()


def findfaces(directory, filename):
    print(filename)
    pixels = pyplot.imread(directory)

    detector = MTCNN()

    try:
        faces = detector.detect_faces(pixels)
        draw_boxes(faces, directory, filename)

    except ValueError:
        print('no face detected in image' ' ' + filename)
        pass



#
for filename in os.listdir("/Users/chiaralin/original_photos/sample/"):
    if not filename.startswith('.'):
        findfaces("/Users/chiaralin/original_photos/sample/" + filename, filename)
        count = count + 1

