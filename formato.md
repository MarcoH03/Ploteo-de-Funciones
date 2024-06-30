1 -  El comando --animate: recibe True o False sin espacios (--animate:True)
2 - El comando --range: recibe x=(inicio,final);y=(inicio,final);... si en alguno no escribo nada entonce coge el de _range
3 - El comando --step: recibe un float y lo usa como step. Si no esta' uso el de default
4 - el comando --cores: recive un numero, si es menor que el numero de cores -1 lo actualiza si no pone el por default
5 - el formato --kind: toma un string y por default usa uno que no recuerdo 
    [x] --kind :
		—2D: 
		[x]plot, (default)
		[x] scatter
		[x] polar
		—3D:
			—animate:true
			[x] scatter
			[x] polar
			[x] line o plot
			—animate:false:
			[x] surface (default)
			[x] contourf
			[x] scatter3d

6 - /config --range:(inicio,final,step) --core:int --kind2d:[plot,scatter,polar] --kind3d:[surface,scatter,contour] --animate:[True,False] --