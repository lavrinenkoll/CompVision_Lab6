import cv2
import numpy as np
import matplotlib.pyplot as plt


# func for matching images with SIFT and BFMatcher algorithms
def matching (img1, img2):
    # Convert images to grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Equalize the histogram
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)

    # Apply Gaussian Blur
    img1 = cv2.GaussianBlur(img1, (5, 5), 7)

    # Normalize images
    #img1 = cv2.normalize(img1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img2 = cv2.normalize(img2, None, alpha=120, beta=255, norm_type=cv2.NORM_MINMAX)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect key points and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Initialize Matcher
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    # Draw matches
    matching_result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Plot the matching result
    #plt.imshow(matching_result)
    #plt.title('Кількість знайдених ознак: '+str(len(good_matches))+'\nКількість всіх ознак: '+str(len(matches)))
    #plt.show()
    #Show the matching result
    # cv2.imshow('Feature Matching Result', matching_result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Return the number of good matches and all matches
    return len(good_matches), len(matches)


# func for matching objects on images and analyze them
def matching_func (img1_path, img2_path, obj1, obj2):
    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Initialize array for counting matches
    obj1_counts = np.zeros(len(obj2))

    # Draw rectangles around objects
    for i in range(len(obj1)):
        cv2.putText(img1, str(i+1), (obj1[i][0], obj1[i][1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(img1, (obj1[i][0], obj1[i][1]), (obj1[i][0] + obj1[i][2], obj1[i][1] + obj1[i][3]), (0, 0, 255), 2)

    for i in range(len(obj2)):
        cv2.putText(img2, str(i+1), (obj2[i][0], obj2[i][1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(img2, (obj2[i][0], obj2[i][1]), (obj2[i][0] + obj2[i][2], obj2[i][1] + obj2[i][3]), (0, 0, 255), 2)

    plt.imshow(img1)
    plt.title('Зображення після виділення контурів')
    plt.show()
    plt.imshow(img2)
    plt.title('Зображення після виділення контурів')
    plt.show()

    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Initialize array for counting matches for each object in img1
    obj1_connections = []

    # Compare each object in img1 with each object in img2
    for i in range(len(obj1)):
        for j in range(len(obj2)):
            # get coordinates of objects
            x1, y1, w1, h1 = obj1[i]
            x2, y2, w2, h2 = obj2[j]
            try:
                # if objects are too different in size, skip them
                if (w1*h1)/(w2*h2) > 5 or (w1*h1)/(w2*h2) < 0.5:
                    continue
                # count matches
                matches, all = matching(img1[x1:x1 + w1, y1:y1 + h1], img2[x2:x2 + w2, y2:y2 + h2])
                # add matches to array
                obj1_counts[j] += matches
            except Exception as e:
                print(e)

        # find best matches for each object in img1
        temp = obj1_counts.copy()
        # first element is number of object in img1
        max_arr = [i + 1]
        for i in range(0, len(obj1_counts)//3):
            # if there are no matches, stop
            if np.max(obj1_counts) == 0:
                break
            # add number of object in img2 to array
            max_arr.append(np.argmax(obj1_counts)+1)
            # set number of matches to 0
            obj1_counts[np.argmax(obj1_counts)] = 0

        # add array to array of all matches
        obj1_connections.append([max_arr, temp])

        # set number of matches to 0
        obj1_counts = np.zeros(len(obj2))

    # print results
    print("Співпадання між об'єктами на 1 та 2 зображенні:")
    for i in range(len(obj1_connections)):
        print(obj1_connections[i])

    # draw lines between objects on images
    all_obj2=[]
    # create new image with both images
    all_img = np.concatenate((img1, img2), axis=1)
    for i in range(len(obj1_connections)):
        for j in range(1, len(obj1_connections[i][0])):
            x1, y1, w1, h1 = obj1[obj1_connections[i][0][0] - 1]
            x2, y2, w2, h2 = obj2[obj1_connections[i][0][j] - 1]
            all_obj2.append(obj1_connections[i][0][j])
            r = np.random.randint(0, 255)
            g = np.random.randint(0, 255)
            b = np.random.randint(0, 255)
            cv2.line(all_img, (x1+w1//2, y1+h1//2), (x2+img1.shape[1]+w2//2, y2+h2//2), (r,g,b), 2)
            cv2.rectangle(all_img, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 1)
            cv2.rectangle(all_img, (x2+img1.shape[1], y2), (x2+img1.shape[1]+w2, y2+h2), (0, 0, 255), 1)

    # show image with lines
    plt.imshow(all_img)
    plt.title("Зображення з лініями між об'єктами")
    #all_img = cv2.resize(all_img, (0, 0), fx=0.5, fy=0.5)
    #cv2.imshow('All', all_img)
    #cv2.waitKey(0)
    plt.show()

    # print results how many times each object on img2 was mentioned in matching with objects on img1
    print("Об'єкти на 2 зображенні згадані разів:")
    unique, counts = np.unique(all_obj2, return_counts=True)
    for value, count in zip(unique, counts):
        print(f"{value}: {count}")

    # draw rectangles around objects on img2, which were mentioned more than once
    img2 = cv2.imread(img2_path)
    for i in range(len(unique)):
        if counts[i] > 1:
            x2, y2, w2, h2 = obj2[unique[i] - 1]
            cv2.putText(img2, str(unique[i]), (x2, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(img2, (x2, y2), (x2+w2, y2+h2), (0, 0, 255), 2)

    # show image with rectangles
    plt.imshow(img2)
    plt.title("Зображення з обраними об'єктами")
    plt.show()
