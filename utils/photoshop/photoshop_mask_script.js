var batPath = "C:/Programming/Bubbles/utils/photoshop/";
var batPath = batPath + "photoshop_to_python.bat"; 


var docName = app.activeDocument.name.replace(/\..+$/, ''); 
var psdPath= String(app.activeDocument.path);
savePath= psdPath.replace("images","masks") +  "/" + docName + ".png";
//alert(savePath);


app.doAction('dataset', 'data');
sfwPNG24(File(savePath));  
 
$.setenv("DOC_NAME", savePath);
File(batPath).execute(); 



app.activeDocument.close(SaveOptions.DONOTSAVECHANGES);







function sfwPNG24(saveFile){  
var pngOpts = new ExportOptionsSaveForWeb;   
pngOpts.format = SaveDocumentType.PNG  
pngOpts.PNG8 = true;   
pngOpts.transparency = false;   
pngOpts.interlaced = false;   
pngOpts.quality = 100;  
activeDocument.exportDocument(saveFile,ExportType.SAVEFORWEB,pngOpts);  
}  