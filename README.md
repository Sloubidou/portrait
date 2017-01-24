TEST DOM

# portrait
DataChallenge Portrait ML


PREMIERE ETAPE AVANT TOUTE MODIFICATION:
Update le projet en local en récupérant la dernière version sur github :
git pull

DEUXIEME ETAPE APRES AVOIR FAIT DES MODIFICATIONS: 
	Si jamais ajout ou suppression de fichiers
		si vous voulez ajouter tous les fichiers :
			git add -A
		si fichier spécifique
			git add pathdufile
	sauver son étape de modification:
		git commit -a -m "description rapide du commit"

TROISIEME ETAPE APRES AVOIR VERIFIER LES MODICATIONS ET SEULEMENT SI ELLES SONT OK:
	git push
si jamais vous n'etes plus à jour, refaire la première étape avec git pull. 


CONFLIT 
Si jamais il y a un conflit, ouvrez les fichiers concernés, et regarder là où il y a les caractères spéciaux du type :
<<<<<<<<<<<<<<<<<<<<<<<<<<<< ou >>>>>>>>>>>>>>>>>>>>>>>>>
Demander à celui qui a modifier le meme fichier pour savoir quelle partie garder. SURTOUT NE PAS TOUT SUPPRIMER, COMMIT ET PUSH. 
Bien vérifier avec l'auteur des modifications !


REPORTER DES MODFIS :
Pour revenir à la version du dernier commit ou pull effectuer : 
git stash 
cela sauve els changement en mémoire mais vous reset à la dernière version


