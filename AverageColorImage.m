function A = AverageColorImage(image)
r = mean(mean(double(image(:,:,1))));
g = mean(mean(double(image(:,:,2))));
b = mean(mean(double(image(:,:,3))));

A = [r,g,b];
%" r " + r + " g " + g + " b " + b
% 
% [imgHeight, imgWidth, colours] = size(image);
% r=0.0;
% g=0.0;
% b=0.0;
% 
% imgHeight + " height " 
% imgWidth + " width "
% 
% for x =1:imgWidth
%     for y =1:imgHeight
%        %% x 
%        %% y + " y  indx"
%        %% rtemp= image(x,y,1);
%        %% gtemp= image(x,y,2);
%        %% btemp= image(x,y,3);
%        
%         r = mean(mean(image(:,:,1));
% 
%         r=r+rtemp;
%         g=g+gtemp;
%         b=b+btemp;
%         
%         
%     end
% end
% 
% totalPixels=imgHeight*imgWidth;
% 
% r=r/totalPixels;
% g=g/totalPixels;
% b=b/totalPixels;
% 
% %% mean(reshape(img,[],3),1);
% %% mean(mean(img));
end
