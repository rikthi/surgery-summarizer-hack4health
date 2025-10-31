const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('desktopApp', {
  isElectron: true
});
