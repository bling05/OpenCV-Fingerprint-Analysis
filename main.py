import os
import cv2 as cv # Importing the OpenCV library

# Converts the sample file into an image to be used later on.
sample_image = cv.imread(f"Samples/sample.BMP")

kp1 = kp2 = mp = None
best_percent = 0
best_file_name = None
best_image = None

# Using SIFT (Scale Invariant Fourier Transform), creates a map of keypoints in both 
# the sample and the potential match's images.
sift = cv.SIFT_create() 
sample_keypoints, sample_descriptors = sift.detectAndCompute(sample_image, None)

# Loops through EVERY file in the database
for file in os.listdir("Database/Real"):

    # Using each file directory, generates an image.
    file_image = cv.imread(f"Database/Real/{file}") 

    file_keypoints, file_descriptors = sift.detectAndCompute(file_image, None)
    print(file)

    # Creates a FLANN matcher using the KD tree algorithm and finds nearest neighbors for keypoints.
    flann_matcher = cv.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}) 
    # Applies keypoint descriptors to matches made by FLANN.
    matches = flann_matcher.knnMatch(sample_descriptors, file_descriptors, k=2) 

    THRESHOLD = 0.15
    # Adds the key point to the list and considers it relevant ONLY IF both keypoints are within 
    # a certain threshold distance from a certain point.
    # In short, if the key points of the sample and current file match, adds the key point to a list to work with later.
    match_keypoints = [sample_kp for (sample_kp, file_kp) in matches if (sample_kp.distance < THRESHOLD * file_kp.distance)]

    # Finds the total # of keypoints in the image with fewer keypoints (using the minimum function)
    total_keypoints = min([sample_keypoints, file_keypoints], key=len)
    # Then calculates the percentage of keypoints that matched.
    percent_match = len(match_keypoints) / len(total_keypoints)

    # If this file is the best match found so far, saves this file's information as the new best 
    if percent_match > best_percent:
        best_percent = percent_match
        best_file_name = file
        best_image = file_image
        kp1, kp2, mp = sample_keypoints, file_keypoints, match_keypoints

print(f"Matching fingerprint: {best_file_name}")
print(f"Percentage match: {best_percent*100}%")
match = cv.drawMatches(sample_image, kp1, file_image, kp2, mp, None)
resized_match = cv.resize(match, None, fx=4, fy=4)
cv.imshow("Match", resized_match)
cv.waitKey(0)
cv.destroyAllWindows()