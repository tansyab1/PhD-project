// import libraries for SQL connection
const mysql = require("mysql");

// function to find the video with the same name as input video
function findVideo(videoName) {
    for (let i = 0; i < videoSource.length; i++) {
        if (videoSource[i].includes(videoName)) {
            return i;
        }
    }
    return -1;
}

//  function to put two videos to two different divs
function putTwoVideos(videoPath1, videoPath2) {
    let videoName1 = getVideoName(videoPath1);
    let videoName2 = getVideoName(videoPath2);
    let videoNum1 = findVideo(videoName1);
    let videoNum2 = findVideo(videoName2);
    if (videoNum1 != -1 && videoNum2 != -1) {
        let video1 = document.getElementById("video1");
        let video2 = document.getElementById("video2");
        video1.setAttribute("src", videoSource[videoNum1]);
        video2.setAttribute("src", videoSource[videoNum2]);
        video1.autoplay = true;
        video2.autoplay = true;
        video1.load();
        video2.load();
    }
}

// function to load all of video inside the folder using JQuery and return the list of video paths
function loadVideos() {
    let videoPaths = new Array();
    $.ajax({
        url: "/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/UnevenIllumination/src/SubTest_GUI/videos/ref_videos/",
        success: function (data) {
            $(data).find("a:contains(.mp4)").each(function () {
                let filename = this.href.replace(window.location.host, "").replace("http:///", "");
                videoPaths.push(filename);
            });
        }
    });
    return videoPaths;
}

// function to suffle the list of video paths
function shuffleVideos(videoPaths) {
    let currentIndex = videoPaths.length, temporaryValue, randomIndex;
    while (0 !== currentIndex) {
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex -= 1;
        temporaryValue = videoPaths[currentIndex];
        videoPaths[currentIndex] = videoPaths[randomIndex];
        videoPaths[randomIndex] = temporaryValue;
    }
    return videoPaths;
}

// function to get the name of video from the path
function getVideoName(videoPath) {
    let videoName = videoPath.split("/").pop();
    return videoName;
}

// function to get the current video name
function getCurrentVideoName() {
    let video1 = document.getElementById("video1");
    let videoName = getVideoName(video1.getAttribute("src"));
    return videoName;
}

// function to play the next video in the list
function playNextVideo() {
    let currentVideoName = getCurrentVideoName();
    let currentVideoNum = findVideo(currentVideoName);
    if (currentVideoNum != -1) {
        if (currentVideoNum < videoSource.length - 1) {
            putTwoVideos(videoSource[currentVideoNum], videoSource[currentVideoNum + 1]);
        }
    }
}

// function to play the previous video in the list
function playPreviousVideo() {
    let currentVideoName = getCurrentVideoName();
    let currentVideoNum = findVideo(currentVideoName);
    if (currentVideoNum != -1) {
        if (currentVideoNum > 0) {
            putTwoVideos(videoSource[currentVideoNum - 1], videoSource[currentVideoNum]);
        }
    }
}

// function to get the value of from the drop down list
function getDropDownValue() {
    let dropDown = document.getElementById("dropDown");
    let value = dropDown.options[dropDown.selectedIndex].value;
    return value;
}

// function to save the value of drop down list to a dictionary with key is the video name
function saveDropDownValue() {
    let currentVideoName = getCurrentVideoName();
    let value = getDropDownValue();
    dropDownValue[currentVideoName] = value;
}

// function to connect to the MySQL database
function connectToDatabase() {
    let db = mysql.createConnection({
        host: "	sql211.epizy.com",
        user: "epiz_32678624",
        password: "qIJQmJDo93MCat",
        database: "epiz_32678624_subTest_results"
    });
    // if the connection is successful, let the user know
    db.connect(function (err) {
        if (err) {
            console.log("Error connecting to Db");
            return;
        }
        console.log("Connection established");
    });
    return db;
}

// function to get the data from table "Test" in the database
function getTableData(db) {
    let sql = "SELECT * FROM Test";
    db.query(sql, function (err, result) {
        if (err) throw err;
        console.log(result);
    });
}

// function to insert the data to table "Test" in the database with 4 columns: ID, name, value, and comment
function insertTableData(db, name, value, comment) {
    let sql = "INSERT INTO Test (name, value, comment) VALUES ?";
    let values = [[name, value, comment]];
    db.query(sql, [values], function (err, result) {
        if (err) throw err;
        console.log("Number of records inserted: " + result.affectedRows);
    });
}

// function to close the connection to the database
function closeConnection(db) {
    db.end();
}