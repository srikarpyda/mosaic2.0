function A = AverageColorImage(image)
r = mean(mean(double(image(:,:,1))));
g = mean(mean(double(image(:,:,2))));
b = mean(mean(double(image(:,:,3))));

A = [r,g,b];

end
