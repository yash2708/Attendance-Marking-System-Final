import takeImage
import takeAttendance
import URLS
import os
import trainImage
haarcasecade_path_face = URLS.get_directory()+"haarcascade_frontalface_default.xml"
trainimage_path = URLS.get_directory()+"TrainingImages"
studentdetail_path = URLS.get_directory()+"StudentDetails\\studentdetails.csv"
attendance_path = URLS.get_directory()+ "Attendance"

def main():
    print("--------Class Attendance Marking System--------")
    print(" 1. Add New Student ")
    print(" 2. Add New Subject ")
    print(" 3. Take Attendance ")
    print(" 4. Train Model ")
    choice = int(input())
    if choice == 1:
        takeImage.TakeImage(0,haarcasecade_path_face,trainimage_path)
    elif choice == 2:
        sub = input("Enter Subject Name:   ")
        path=os.path.join(attendance_path,sub)
        try:
            os.mkdir(path)
        except:
            print("Subject already Exists")
    elif choice == 3:
        takeAttendance.attendance(haarcasecade_path_face,studentdetail_path,attendance_path)
    elif choice == 4:
        trainImage.TrainImage_LBHP_Softmax(trainimage_path)

    print(choice)

if __name__ =="__main__":
    main()