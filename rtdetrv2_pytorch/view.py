import cv2
import sys


def main(args):
    img = cv2.imread(args[1])

    cv2.imshow("image", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main(sys.argv)
