function [Optimask] =  generateM(filename, maskFolder, formatFolder, OpticalZmask, OpticalNx, OpticalNy, OpticalNablax, OpticalNablay, sensingmatrix, bernouillip, bernoulliseed)

if strcmp(sensingmatrix, 'hadamard')
    Optimask = hadamard(OpticalNx*OpticalNy) < 0;
elseif strcmp(sensingmatrix, 'bernouilli')
    rng(bernoulliseed) %if nargin
    Optimask = rand(OpticalNx*OpticalNy) < bernouillip;
end
imshow(Optimask)
reshaped = @(x) reshape(x, [OpticalNx*OpticalNy, 1]);
unreshape = @(x) reshape(x, [OpticalNx, OpticalNy]);
x = -(OpticalNx-1)/2*OpticalNablax:OpticalNablax:(OpticalNx-1)/2*OpticalNablax;
y = -(OpticalNy-1)/2*OpticalNablay:OpticalNablay:(OpticalNy-1)/2*OpticalNablay;
[x,y]= meshgrid(x,y);
mkdir(maskFolder);
p1mask = (min(x(1,:)) - OpticalNablax/2);
p2mask = (max(x(1,:)) + OpticalNablax/2);
p1scene = (min(x(1,:)) - OpticalNablax/2) * 2;
p2scene = (max(x(1,:)) + OpticalNablax/2) * 2;
disp('Generating grasp configuration files:')
for i = 1:length(Optimask)
    graspconfig = compose('%s/%s%s', maskFolder,sensingmatrix,string(i));
    copyfile(formatFolder, graspconfig{1});
    disp(graspconfig{1});
    x = reshaped(x);
    y = reshaped(y);
    z = reshaped(OpticalZmask * ones(OpticalNx,OpticalNy));OpticalNablax = OpticalNablax - 0.000000;OpticalNablay = OpticalNablay - 0.000000;
    mask = [reshaped(1:OpticalNx*OpticalNy),x-OpticalNablax,y-OpticalNablay,z,x-OpticalNablax,y,z,x,y,z];  
    mask = mask(Optimask(i,:),:);
    updatefile = compose('%s/%s%s/%s', maskFolder,sensingmatrix, string(i), filename);
    fileID = fopen(updatefile{1},'a');
%     fprintf(fileID,'\nplanargridonmask  planar_grid  \n(\n  frequency        : ref(THz750),\n  near_dist        : 0.1 m,\n  x_range          : struct(start: %f, end: %f, np: 64, unit: m),\n  y_range          : struct(start: %f, end: %f, np: 64),\n  file_name        : planargridonmask.grd\n)', [p1mask,p2mask,p1mask,p2mask]);
%  
%     fprintf(fileID,'\nplanargridonscene  planar_grid  \n(\n  frequency        : ref(THz750),\n  near_dist        : 0.19 m,\n  x_range          : struct(start: %f, end: %f, np: 128, unit: m),\n  y_range          : struct(start: %f, end: %f, np: 128),\n  file_name        : planargridonscene.grd\n)', [p1scene,p2scene,p1scene, p2scene]);
%  
%     fprintf(fileID, '\nplanargridfromscene  planar_grid  \n(\n  frequency        : ref(THz750),\n  near_dist        : 0.1 m,\n  x_range          : struct(start: %f, end: %f, np: 64, unit: m),\n  y_range          : struct(start: %f, end: %f, np: 64),\n  file_name        : planargridfromscene.grd\n)', [p1scene, p2scene, p1scene, p2scene]);
%     fprintf(fileID,"recatngularshape  rectangular_plate\n(\n  coor_sys         : ref(coordsystem),\n  corner_1         : struct(x: %f m, y: %f m, z: %f m),\n  corner_2         : struct(x: %f m, y: %f m, z: %f m),\n  opp_point        : struct(x: %f m, y: %f m, z: %f m)\n)\n", [p1scene,p1scene,0.19,p1scene,p2scene,0.19,p2scene,p2scene,0.19]);
    fprintf(fileID,"\nmask%d  rectangular_plate\n(\n  coor_sys         : ref(coordsystem),\n  corner_1         : struct(x: %f m, y: %f m, z: %f m),\n  corner_2         : struct(x: %f m, y: %f m, z: %f m),\n  opp_point        : struct(x: %f m, y: %f m, z: %f m)\n)\n", mask');
    fprintf(fileID,"maskcluster  scatterer_cluster  \n(\n  scatterers       : sequence(");
    fprintf(fileID,"ref(mask%d),",mask(:,1));
    fprintf(fileID,")\n)");
    fclose(fileID);
end
Optimask = Optimask == 0;
heatmap(double(unreshape(Optimask(randi(OpticalNx*OpticalNy),:)))); %if nargin
end

