# Passos para execução do dinoAI

1. Instalar python3
sudo apt-get install python3

2. Instalar pygame
pip install pygame

3. Executar o jogo
python3 dinoAI.py

Como tirar a interface gráfica:

Na linha 11 existe uma flag chamada RENDER_GAME basta trocar ela para False,
assim o jogo roda sem interface gráfica ficando mais rapido pois o jogo
renderiza em 60 fps com a tela, sem a tela o fps é ilimitado.

RENDER_GAME = False

Na linha 10, há uma variável GAME_MODE que permite alterar o modo de jogo, dando controle a um jogador humano.

# Como treinar vários Dinos ao mesmo tempo.

User o código de dinoAIParallel.py, as principais modificações são:

- A função playGame recebe como parametro as soluções para o seu problema
- A função playGame retorna não apenas uma pontuação mas um vetor de pontuação para cada solução.
- Existe a função manyPlaysResultsTrain que recebe as soluções joga com elas X rounds e retorna a media de cada um da população
- Existe a função manyPlaysResultsTest que recebe apenas uma solução e retorna a media de pontuação do individuo

OBS: Ao usar esse código vários dinossauros aparecem na tela jogando ao mesmo tempo, para diferencia-los existe um
quadrado colorido em volta de cada dino, com uma cor diferente para cada dino.


O dinoAIParallel.py não foi implementado.





