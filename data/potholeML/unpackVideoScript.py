import numpy as np
import cv2
import os

# FILE CONSTANTS USED FOR OS DIRECTORY TRAVERSING AND FILE GENERATION
VIDEO_FOLDER = "Videos/"
POTHOLE_FOLDER = "Pothole/"
NOTPOTHOLE_FOLDER = "notPothole/"
VIDEO_EXT = ".mp4"
IMG_EXT = ".png"
INPUT_HANDLER = {'Y':POTHOLE_FOLDER,'N':NOTPOTHOLE_FOLDER}
FOLDER_NAME = (("" + VIDEO_FOLDER + POTHOLE_FOLDER, "Images/" + POTHOLE_FOLDER),("" + VIDEO_FOLDER + NOTPOTHOLE_FOLDER,"Images/"+NOTPOTHOLE_FOLDER))

def main():
    seperateVideoFrames()

def seperateVideoFrames():
    # ITERATE THROUGH GIVEN VIDEO FOLDERS STORED IN TUPLE CONSTANT FOLDER_NAME
    for folder in FOLDER_NAME:
        # ITERATE THROUGH VIDEOS WITHIN CURRENT FOLDER folder[0]
        for filename in os.listdir(folder[0]):
            # CHECK IF VIDEO IS ALREADY PROCESSED
            image_name = filename[:-4] + "_0" + IMG_EXT
            if(image_name in os.listdir(folder[1])):
                print("Video ( %s ) already processed into individual frames, please rename video file and try again" %(filename))

            # SEPERATE VIDEO KNOWING VIDEO NOT ALREADY PROCESSED
            elif(filename.endswith(VIDEO_EXT)):
                # DEBUG SATEMENTS
                print("Unpacking video ( %s ) into individual frames into %s." %(filename, folder[1]))
                # print(filename[:-4])
                # LOAD AND CAPTURE VIDEO FROM FOLDER
                video_file = "" + folder[0] + filename
                capture = cv2.VideoCapture(video_file)
                # INIT COUNTER FOR INDIVIDUAL FRAME NAMING
                counter = 0
                # DEBUG STATEMENT FOR VIDEO READING
                print("Opened!" if capture.isOpened() else "Closed!")
                while(capture.isOpened()):
                    # GENERATE IMAGE FRAME NAME FOR SAVING FRAME TO CORRECT CLASSIFIED IMAGE FOLDER
                    image_name = "" + folder[1] + filename[:-4] + "_" + str(counter) + IMG_EXT
                    # CAPTURE VIDEO FRAME
                    return_val, frame = capture.read()
                    # CHECK TO SEE IF END OF VIDEO
                    if(return_val):
                        # IF NOT END OF VIDEO
                        # DEBUG STATEMENT FOR CHECKING CORRECTNESS
                        # print(image_name)
                        status = cv2.imwrite(image_name,frame)
                        counter = counter + 1
                    else:
                        # IF END OF VIDEO, BREAK FROM LOOP AND START NEXT ITERATION
                        print("...FINISHED!")
                        break
                # RELEASE VIDEO CAPTURE USED
                capture.release()

# CALL MAIN FUNCTION
main()
