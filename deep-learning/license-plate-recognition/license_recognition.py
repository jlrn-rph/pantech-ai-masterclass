# LICENSE PLATE RECOGNITION WITH OPENALPR CLOUD

# import libraries
import cv2, time, json, base64, requests, csv
from authKey import SECRET_KEY

# initialize and read camera
cam = cv2.VideoCapture(1)
while True:
    _, img = cam.read()
    cv2.imshow('license plate', img) # display windows
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        cv2.destroyAllWindows() # destroy windows when 'q' is pressed 
        print('Captured...')
        cv2.imwrite('cap_img.jpg', img) # save image
        time.sleep(5.0)
        IMAGE_PATH = 'cap_img.jpg' # path of saved image

        with open(IMAGE_PATH, 'rb') as image_file: # load image as binary
            img_base64 = base64.b64encode(image_file.read()) # encode the image

        url = 'https://api.openalpr.com/v2/recognize_bytes?recognize_vehicle=1&country=us&secret_key=%s' % (SECRET_KEY) # openalpr api url
        req = requests.post(url, data = img_base64) # post the image

        num_plate = json.dumps(req.json(), indent = 2) # get image in json form

        # slice the plate number only
        info = list(num_plate.split('candidates')) 
        # print(info)
        plate = info[1].split(',')[0:3]
        p = plate[1].split(':')
        number = p[1].replace('"', '').lstrip()
        print(number)

        with open('deep-learning/license-plate-recognition/license_plates.csv', 'r', newline = '') as csvFile: # read csv file
            reader = csv.DictReader(csvFile) 
            # if plate number is found, read the user data
            for row in reader:
                if number == row['plate_num']:
                    print('Owner name: {}'.format(row['name']))
                    print('Plate number: {}'.format(row['plate_num']))
                    print('State: {}'.format(row['state']))
                    break
                
                # if plate number is not found, add the user data
                elif number != row['plate_num']:
                    print('DATA NOT FOUND')
                    name = input('Enter your name: ')
                    state = input('Enter your state: ')
                    with open('deep-learning/license-plate-recognition/license_plates.csv', 'a', newline = '') as csvFile: # append to csv file
                        writer = csv.writer(csvFile)
                        writer.writerow([number, name, state])
                    break

    # close windows when 'esc' is pressed 
    elif key == 27:
        break

cam.release()
cv2.destroyAllWindows()

