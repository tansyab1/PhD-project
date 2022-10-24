# python main file to run the application 

from PyQt5 import QtCore, QtMultimedia
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys
import csv
import os

# load libraries data
from utils.data import load_data, get_reference_video
# load ui file from utils folder
from utils.camera_update import Ui_Camera

class DictionaryTableModel(QtCore.QAbstractTableModel):
    def __init__(self, data, headers):
        super(DictionaryTableModel, self).__init__()
        self._data = data
        self._headers = headers

    def data(self, index, role):
        if role == Qt.DisplayRole:
            # Look up the key by header index.
            column = index.column()
            column_key = self._headers[column]
            return self._data[index.row()][column_key]

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

    def columnCount(self, index):
        # The length of our headers.
        return len(self._headers)

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._headers[section])

            if orientation == Qt.Vertical:
                return str(section)

# class for iterating through the distorted videos
class Iterator:
    def __init__(self, path):
        self.iterator = iter(path)
        self.current = None
    def __next__(self):
        try:
            self.current = next(self.iterator)
        except StopIteration:
            self.current = None
        finally:
            return self.current

# class main window to load UI 
class MainWindow(QMainWindow):
    def __init__(self, distorted_video_iter, distorted_video_paths, reference_video_paths, distortion_type_iter):
        super(MainWindow, self).__init__()
        self.ui = Ui_Camera()

        self.playerRef = QtMultimedia.QMediaPlayer(None, QtMultimedia.QMediaPlayer.VideoSurface)
        self.playerDist = QtMultimedia.QMediaPlayer(None, QtMultimedia.QMediaPlayer.VideoSurface)

        self.ui.setupUi(self)
        self.distorted_video_iter = distorted_video_iter
        self.reference_video_paths = reference_video_paths
        self.distortion_type_iter = distortion_type_iter
        self.templist = distorted_video_paths
        self.is_finished = False
        self.is_valid = False
        self.process_start = True

        self.nextvideo()
        self.process_start = False

        # list of the processed distorted videos
        self.processed_distorted_videos = []
        self.previous_distorted_video = self.get_current_distorted_video()
        # show the dictionary in the tableView 
        self.headers = ['Name', 'Value']

        # dictionary of the distorted videos and the corresponding values
        self.distorted_videos_values = []
        self.current_distorted_video = self.get_current_distorted_video()

        # setting the tableView with two columns. The first column is "Distorted Video" and the second column is "Value"
        # self.ui.tableView.setColumnCount(2)
        # self.ui.tableView.setHorizontalHeaderLabels(["Distorted Video", "Value"])

        # if the button start is clicked, run the function startvideo
        self.ui.start.clicked.connect(self.startvideo)

        # if the button stop is clicked, run the function stopvideo
        self.ui.stop.clicked.connect(self.stopvideo)

        # if the button next is clicked, run the function nextvideo
        self.ui.next.clicked.connect(self.nextvideo)

        # if the button previous is clicked, run the function previousvideo
        # self.ui.previous.clicked.connect(self.previousvideo)
        
        # if the button validate is clicked, run the function validate
        self.ui.valide.clicked.connect(self.validate)

        # if the button save is clicked, run the function save
        self.ui.save.clicked.connect(self.save)

        # if the slider is moved, run the function update_slider_value
        self.ui.verticalSlider.valueChanged.connect(self.update_slider_value)

        self.playerRef.setVideoOutput(self.ui.referencevideo)
        self.playerDist.setVideoOutput(self.ui.distortedvideo)

        

    def closeEvent(self, event):
        print ("User has clicked the red x on the main window")
        event.accept()



    # define function validate to validate the two videos
    def validate(self):
        self.is_valid = True
        if self.is_finished == False:
            verticalSlider_value = self.get_slider_value()
            # get the current distorted video
            self.current_distorted_video = self.get_current_distorted_video()
            

            # check the QCheckBox
            self.ui.checkRef.setChecked(True)
            self.ui.checkDist.setChecked(True)

            # check if the current distorted video is already in the dictionary. If not, add it to the dictionary, 
            # else update the value of the current distorted video in the dictionary
            if self.check_if_already_processed():
                for i in self.distorted_videos_values:
                    if i["Name"] == self.current_distorted_video:
                        i["Value"] = verticalSlider_value
                # show the notification message
                self.model = DictionaryTableModel(self.distorted_videos_values, self.headers)
                self.ui.tableView.setModel(self.model)
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText("The value of the current distorted video has been updated")
                msg.setWindowTitle("Notification")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            else:
                self.processed_distorted_videos.append(self.current_distorted_video)
                self.distorted_videos_values.append({"Name": self.current_distorted_video, "Value": verticalSlider_value})
                self.model = DictionaryTableModel(self.distorted_videos_values, self.headers)
                self.ui.tableView.setModel(self.model)
                
    # define a funtion check if the dictionary has already the current distorted video
    def check_if_already_processed(self):
        for i in self.processed_distorted_videos:
            if i == self.current_distorted_video:
                return True
        return False

    # define function nextvideo
    def nextvideo(self):
        # check the video is valid or not
        if self.is_valid == False and self.process_start == False:
            # show the message box to ask the user to validate the video
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please validate the video")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        # load video from videos iterator and put them in the two windows referencevideo and distortedvideo
        self.distorted_video = next(self.distorted_video_iter)
        self.distortion_type = next(self.distortion_type_iter)
        if self.distorted_video is None:
            # show a message box to inform the user that all the videos have been processed
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("All the videos have been processed")   
            msg.setWindowTitle("Information")
            msg.exec_()
            self.is_finished = True
            return

        self.reference_video = get_reference_video(self.distorted_video, self.reference_video_paths)
        self.playerRef.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(self.reference_video)))
        self.playerDist.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(self.distorted_video)))
        self.current_distorted_video = self.get_current_distorted_video()
        # stop two videos
        self.stopvideo()

        # show the name of the current distorted video in the QTextEdit
        self.set_text()
        self.is_valid = False

        # uncheck the QCheckBox
        self.ui.checkDist.setChecked(False)
        self.ui.checkRef.setChecked(False)

    # define function previousvideo
    def previousvideo(self):
        # get current distorted video
        self.current_distorted_video = self.get_current_distorted_video()
        # get index of the previous distorted video in the distorted videos iterator
        index_video = self.templist.index(self.current_distorted_video)
        # get the previous distorted video
        if index_video > 0:
            self.previous_distorted_video = self.templist[index_video - 1]
        else:
            self.previous_distorted_video = self.templist[index_video]

        self.distorted_video = self.previous_distorted_video
        self.reference_video = get_reference_video(self.distorted_video, self.reference_video_paths)

        self.playerRef.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(self.reference_video)))
        self.playerDist.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(self.distorted_video)))

        # self.playerRef.setVideoOutput(self.ui.referencevideo)
        # self.playerDist.setVideoOutput(self.ui.distortedvideo)

        # get the current distorted video and previous distorted video
        self.current_distorted_video = self.get_current_distorted_video()
        
        # stop the two videos
        self.stopvideo()

        # show the name of the current distorted video in the QTextEdit
        self.set_text()
        self.is_valid = False
        # check if the current distorted video is in the processed distorted videos list, if yes check the QCheckBox
        if self.current_distorted_video in self.processed_distorted_videos:
            self.ui.checkDist.setChecked(True)
            self.ui.checkRef.setChecked(True)

        
    # define function startvideo to start the two videos
    def startvideo(self):

        # if the distorted video is different to the last item of the processed distorted videos list, add the current distorted video to the list
        
        # if len(self.processed_distorted_videos) > 1:
        #     if self.current_distorted_video != self.processed_distorted_videos[-1]:
        #         self.processed_distorted_videos.append(self.current_distorted_video)

        # play the two videos from the current position of the videos
    
        self.playerRef.play()
        self.playerDist.play()


    # define function stopvideo to stop the two videos  
    def stopvideo(self):

        self.playerRef.pause()
        self.playerDist.pause()

        # set the text to TextEdit
    def set_text(self):
        self.ui.namedist.setText("Distortion: "+self.distortion_type)
        self.ui.nameref.setText("Reference: " + os.path.basename(self.reference_video))

    # define the function to set videoOutput
    # def set_videoOutput(self):
    #     self.playerRef.setVideoOutput(self.ui.referencevideo)
    #     self.playerDist.setVideoOutput(self.ui.distortedvideo)



    # get current distorted video which is playing
    def get_current_distorted_video(self):
        return self.distorted_video_iter.current

    # get value of the slider
    def get_slider_value(self):
        return self.ui.verticalSlider.value()

    # update the value of the slider when the slider is moved to the lcdNumber
    def update_slider_value(self):
        self.ui.lcdNumber.display(self.get_slider_value())

    # save the dictionary of the distorted videos and the corresponding values in a csv file
    def save(self):
        #get the name of the observer from the QLineEdit
        observer_name = self.ui.name.text()

        # check the RadioButtons of expert or non-expert
        if self.ui.expert.isChecked():
            is_expert = "expert"
        else:
            is_expert = "non-expert"

        # save file QFileDialog with the filename is expert or non-expert and the name of the observer
        filename = QFileDialog.getSaveFileName(self, 'Save File', f"{is_expert}_{observer_name}.csv", "CSV(*.csv)")
        with open(filename[0], 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'value'])
            # save the list of the dictionary in the csv file
            for item in self.distorted_videos_values:
                writer.writerow([item["Name"], item["Value"]])

        # show the QMessageBox
        print("File saved")
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("The file has been saved, thank you for your participation!")
        # show the QMessageBox
        msg.exec_()
        

if __name__ == '__main__':
    # load the reference videos and distorted videos
    reference_video_paths, distorted_video_paths, distortion_type = load_data(os.path.dirname(__file__)+'/videos')
    
    # convert the distorted_video_paths to iterator by using class Interator
    distorted_video_iter = Iterator(distorted_video_paths)
    distortion_type_iter = Iterator(distortion_type)

    app = QApplication(sys.argv)
    window = MainWindow(distorted_video_iter,distorted_video_paths, reference_video_paths, distortion_type_iter)
    window.show()
    sys.exit(app.exec_())

