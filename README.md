© Copyright — Este projeto é propriedade exclusiva de Tomás Serra. Todos os direitos reservados. Todo o projeto Kasparov está patenteado em nome de Tomás Serra.

# ♟️ Kasparov V2

**Sistema de Xadrez Inteligente com Visão por Computador**

Kasparov é um sistema que deteta um tabuleiro de xadrez físico e as respetivas peças em tempo real, através de uma câmara de profundidade, converte o estado do tabuleiro em notação FEN, e integra esse estado com um motor de xadrez de nível profissional — oferecendo sugestões de jogadas, avaliação de posição e feedback por voz em três idiomas.

Projeto desenvolvido por **Tomás Serra** no âmbito da PAP (Prova de Aptidão Profissional) do curso profissional de Informática — Sistemas.

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Funcionalidades](#-funcionalidades)
- [Arquitetura e Tecnologias](#-arquitetura-e-tecnologias)
- [Requisitos de Hardware](#-requisitos-de-hardware)
- [Instalação](#-instalação)
- [Como Correr o Projeto](#-como-correr-o-projeto)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Capturas de Ecrã / Demo](#-capturas-de-ecrã--demo)
- [Licença e Patente](#-licença-e-patente)
- [Autor](#-autor)

---

## 🎯 Visão Geral

O Kasparov combina **visão por computador**, **deteção de profundidade** e **motores de xadrez** para transformar um tabuleiro físico num tabuleiro digital jogável:

1. A câmara **Intel RealSense D415** captura imagem RGB + profundidade do tabuleiro.
2. Um modelo **YOLOv11** deteta as peças e as suas posições.
3. A lógica de mapeamento (`GridModel`) converte as coordenadas da câmara em casas do tabuleiro (A1–H8), tendo em conta orientação e lado do jogador.
4. O estado do tabuleiro é convertido em **notação FEN**.
5. O motor **Stockfish (NNUE)** analisa a posição e devolve avaliação e melhores jogadas.
6. Tudo é apresentado numa interface gráfica em **PySide6**, com narração por voz (TTS).

---

## ✨ Funcionalidades

- 🎥 **Deteção física de peças em tempo real** via YOLOv11 + RealSense D415
- ♟️ **Mapeamento automático para FEN**, incluindo lógica de orientação do tabuleiro (lado das brancas / pretas)
- 🧠 **Motor de análise Stockfish/NNUE** integrado
- 🎓 **Modo Treino** — sugestão das 3 melhores jogadas com avaliação em centipawns
- 🗣️ **Text-to-Speech trilingue** (Português, Inglês, Francês) via `edge-tts`, com anúncio de jogadas
- 🖥️ **Interface gráfica moderna** em PySide6 (tema "deep space / holográfico")
- 🐕 **Watchdog timer** para monitorização de estado e falhas do sistema
- 🔒 Lógica de *board locking* com prevenção de condições de corrida

---

## 🏗️ Arquitetura e Tecnologias

| Componente | Tecnologia |
|---|---|
| Linguagem | Python |
| Interface gráfica | PySide6 |
| Deteção de objetos | YOLOv11 |
| Câmara | Intel RealSense D415 (RGB + Profundidade) |
| Motor de xadrez | Stockfish (rede NNUE) |
| Síntese de voz | edge-tts (vozes neuronais PT/EN/FR) |
| Representação de estado | FEN (Forsyth–Edwards Notation) |

---

## 🖥️ Requisitos de Hardware

- Câmara **Intel RealSense D415** (obrigatória — o projeto funciona, por enquanto, exclusivamente com este modelo)
- Tabuleiro e peças de xadrez físicas compatíveis com o dataset de deteção
- Computador com suporte a Python 3.x e, idealmente, GPU compatível com CUDA para inferência mais rápida do YOLOv11

---

## ⚙️ Instalação

```bash
# 1. Clonar o repositório
git clone https://github.com/<utilizador>/kasparov_v.2.git
cd kasparov_v.2

# 2. Criar e ativar ambiente virtual
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Instalar o SDK da Intel RealSense
# https://github.com/IntelRealSense/librealsense
```

> 📌 Certifica-te de que os drivers da RealSense D415 estão corretamente instalados e que a câmara é reconhecida pelo sistema operativo antes de correr o projeto.

---

## ▶️ Como Correr o Projeto

```bash
python main.py
```

1. Liga a câmara Intel RealSense D415 antes de iniciar a aplicação.
2. Posiciona o tabuleiro no campo de visão da câmara.
3. A aplicação irá calibrar a deteção do tabuleiro e das peças.
4. Usa a interface PySide6 para escolher entre modo normal ou **Modo Treino**.
5. As jogadas e avaliações serão anunciadas por voz no idioma configurado.

---

## 📁 Estrutura do Projeto

```
kasparov_v.2/
├── assets/
│   └── pieces/          # Imagens PNG das peças de xadrez (tabuleiro virtual)
├── src/
│   ├── grid_model.py    # Mapeamento câmara → casas do tabuleiro
│   ├── engine/           # Integração com Stockfish/NNUE
│   ├── vision/           # Deteção YOLOv11 + RealSense
│   ├── tts/              # Text-to-Speech trilingue
│   └── gui/               # Interface PySide6
├── main.py
├── requirements.txt
└── README.md
```

> As imagens `.png` das peças devem ser colocadas em `assets/pieces/`, mantendo esta pasta dedicada exclusivamente aos recursos visuais do tabuleiro virtual.

---




## 🔒 Licença e Patente

**Todo o projeto Kasparov (código, design, metodologia e materiais associados) está patenteado em nome de Tomás Serra.**

Qualquer reprodução, distribuição, ou utilização comercial deste projeto sem autorização expressa do autor é proibida. Para questões de licenciamento ou colaboração, contactar diretamente o autor.

---

## 👤 Autor

**Tomás Serra**
Curso Profissional de Informática — Sistemas
Projeto desenvolvido no âmbito da Prova de Aptidão Profissional (PAP)

---

Feito com um tabuleiro de xadrez , café e muita depuração de bugs.
