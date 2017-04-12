function A = AverageColorTile(image, x1, y1, x2, y2)

r = mean(mean(double(image(x1:x2,y1:y2,1))));
g = mean(mean(double(image(x1:x2,y1:y2,2))));
b = mean(mean(double(image(x1:x2,y1:y2,3))));

A = [r,g,b];

end
