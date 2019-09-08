var batPath = "C:/Programming/Bubbles/utils/photoshop/";
var batPath = batPath + "photoshop_to_python.bat"; 

var docName = app.activeDocument.name.replace(/\..+$/, ''); 
var psdPath= String(app.activeDocument.path);
var dirPath= psdPath.replace("images","masks");
savePath= dirPath +  "/" + docName + ".png";
//alert(savePath);

var fold= new Folder(dirPath)
if(!fold.exists)
	{
	fold.create();
	//alert("create");
	}

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