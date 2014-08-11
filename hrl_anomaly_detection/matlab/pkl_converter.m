clc
clear all
close all

%dirName='/home/dpark/svn/robot1_data/usr/advait/ram_www';
dirName='/home/dpark/git/hrl-assistive/hrl_anomaly_detection/matlab/data';

filelist = getAllFiles(dirName);

for elm = filelist'
   [pathstr, name, ext] = fileparts(elm{1});
   if strcmp(ext,'.pkl')==true
      try
        pkl_data = loadpickle(elm{1});
      catch
        continue
      end
      key_names= fieldnames(pkl_data);
      var_names='';
      
      % Get data
      for key_name = key_names'
          eval(strcat(key_name{1},'=','getfield(pkl_data,key_name{1});'));
          if strcmp(var_names,'')==true
            var_names = strcat(' '' ',key_name{1}, ''' ');
          else
            var_names = strcat(var_names, ',', ''' ',key_name{1}, ''' ');              
          end
      end
      
      % Save to mat file
      mat_file = fullfile(pathstr,strcat(name,'.mat'));
      eval(strcat('save(mat_file,', var_names, ')'));
   end
    
end
