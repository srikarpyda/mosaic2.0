function nearestImage = Nearest( rgbValues, imageMap )
%UNTITLED Summary of this function goes here
% TTD: FIGURE OUT HOW TO ACCESS ALL OF THE IMAGES TO BE USED AS TILES
%   Detailed explanation goes here
nearestImage="empty name";

k=keys(imageMap);
v=values(imageMap);

smallest=10000.00;

for i=1:length(k)
    
    
    dist=Distance(rgbValues,v(i));
    
    if(dist<smallest)
        
        smallest=dist;
        nearestImage=k(i);
    end
    
end

%%remove(imageMap,nearestImage);

end

