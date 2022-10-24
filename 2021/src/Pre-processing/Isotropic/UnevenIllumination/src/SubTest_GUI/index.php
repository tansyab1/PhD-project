<?php
// when clicking on the button, the image is processed
if (isset($_POST['buttontest'])) {
    $host = "sql211.epizy.com";
    $user = "epiz_32678624";
    $pass = "qIJQmJDo93MCat";
    $db = "epiz_32678624_subTest_results";

    $con = mysqli_connect($host, $user, $pass, $db);
    if ($con) {
        echo "Connection successful test";
    } else {
        echo "Connection error";
    }
}
