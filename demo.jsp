<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>Insert title here</title>
<link rel="stylesheet" href="css/style.css">
<style type="text/css">
    <%@include file="css/style.css" %>
    </style>
</head>
<marquee> ENTER THE REQUIRED PARAMETERS OF WATER </marquee>
<body>
<div class="container">  
  <form id="contact" action="" method="post">
    <h3>KNOW IF YOUR IS SAFE OR UNSAFE</h3>
    <h4>Contact us for custom quote</h4>
    <fieldset>
      <input placeholder="BOD" type=INT tabindex="3" required autofocus>
    </fieldset>
    <fieldset>
      <input placeholder="DO" type=INT tabindex="3" required>
    </fieldset>
    <fieldset>
      <input placeholder="CONDUCTIVITY" type=INT tabindex="3" required>
    </fieldset>
    <fieldset>
      <button name="submit" type="submit" id="contact-submit" data-submit="...Sending">Submit</button>
    </fieldset>
    
  </form>
</div>
</body>
</html>
