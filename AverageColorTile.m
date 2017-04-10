function A = AverageColorTile(image, x1, y1, x2, y2)

r = mean(mean(double(image(x1:x2,y1:y2,1))));
g = mean(mean(double(image(x1:x2,y1:y2,2))));
b = mean(mean(double(image(x1:x2,y1:y2,3))));

A = [r,g,b];
% 
% [imgHeight, imgWidth, colours] = size(image);
% 
% r=0.0;
% g=0.0;
% b=0.0;
% 
% 
% r = mean(mean(double(image(x1:x2,y1:y2,1))));
% g = mean(mean(double(image(x1:x2,y1:y2,2))));
% b = mean(mean(double(image(x1:x2,y1:y2,3))));
% 
% for y =y1:y2
%     for x =x1:x2
%         
%         rtemp= image(x,y,1);
%         gtemp= image(x,y,2);
%         btemp= image(x,y,3);
%         
%         r=r+rtemp;
%         g=g+gtemp;
%         b=b+btemp;
%         
%     end
% end
% 
% totalPixels=imgHeight*imgWidth;
% 
% 
% r=r/totalPixels;
% g=g/totalPixels;
% b=b/totalPixels;
% 
% %%" r " + r + " g " + g + " b " + b
end
