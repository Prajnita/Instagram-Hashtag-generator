import Taglist
import Prediction
from Taglist import Hashtag


identify_class = Prediction.image_classification('Test/103.jpg')

def getTags(list_of_keywords):
    if identify_class == 1:
        tags = Hashtag['Poledance']

        for key in list_of_keywords:
            for tag in Taglist[key]:
                tags.append(tag)
        print(list(set(tags)))

        return list(set(tags))

    elif identify_class == 0:
        tags = Hashtag['Yoga']

        for key in list_of_keywords:
            for tag in Taglist[key]:
                tags.append(tag)
        print(list(set(tags)))

        return list(set(tags))


def getHashTagString(list_of_tags):
    hashTagString = ""

    for tag in list_of_tags:
        hashTagString += " #" + tag

    return hashTagString


if __name__ == '__main__':
    import sys

    print(getHashTagString(getTags(sys.argv[1:])).strip())
