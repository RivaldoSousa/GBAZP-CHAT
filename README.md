# Farm Animal 3D Web Generator

Interface web para gerar modelos STL de animais de fazenda (Bovino, Equino, Suíno, Ave, Ovino) prontos para impressão 3D.

## Funcionalidades

- **5 espécies**: Bovino, Equino, Suíno, Ave, Ovino
- **Dois modos**:
  - **Sólido**: modelo único (um arquivo STL)
  - **Articulado**: peças separadas com juntas móveis (bola-soquete / pino)
- **Escala ajustável**: padrão 10 mm → modelo ~10×5×7 cm (cabe na mão)
- **Download em ZIP**: arquivos STL organizados + guia de montagem (para articulados)
- **Sem dependências externas** no gerador (apenas Python padrão + Flask para web)

## Como rodar localmente

```bash
# 1. Clone o repositório
git clone https://github.com/dataloggertiete-hub/farm-animal-3d-web.git
cd farm-animal-3d-web/farm_animal_web

# 2. Crie e ative um ambiente virtual
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute a aplicação
python app.py
```

Acesse **http://127.0.0.1:5000** no navegador.

## Deploy

### Render (gratuito)
1. Conecte este repositório no [Render](https://render.com)
2. New → Web Service
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `python app.py`

### Docker
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY farm_animal_web/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY farm_animal_web/ .
EXPOSE 5000
CMD ["python", "app.py"]
```

```bash
docker build -t farm-animal-3d .
docker run -p 5000:5000 farm-animal-3d
```

## Estrutura do projeto

```
farm_animal_web/
├── app.py              # Servidor Flask
├── gerador.py          # Lógica de geração 3D (STL)
├── requirements.txt    # Flask
├── templates/
│   └── index.html      # Formulário web
├── static/             # (opcional) CSS/JS extras
└── generated/          # Arquivos temporários (ignorados no git)
```

## Como funciona

1. Usuário escolhe a espécie, escala e tipo (sólido/articulado)
2. O Flask chama `gerador.py` que cria a geometria paramétrica
3. STL(s) são salvos em pasta temporária única (UUID)
4. Tudo é zipado e entregue como download
5. Pasta temporária pode ser limpa periodicamente

## Exemplos de uso

| Espécie | Escala (mm) | Tamanho aprox. |
|---------|-------------|----------------|
| Bovino  | 10          | 10 × 5 × 7 cm  |
| Equino  | 10          | 10 × 5 × 7 cm  |
| Suíno   | 10          | 10 × 5 × 7 cm  |
| Ave     | 10          | 8 × 4 × 5 cm   |
| Ovino   | 10          | 9 × 4 × 6 cm   |

## Licença

MIT - Use livremente para projetos pessoais, educacionais ou comerciais.

## Autor

Desenvolvido para dataloggertiete-hub
## GBAZP Chat com busca acadêmica e modelo aberto

A interface em `GBAZP-CHAT/index.html` agora usa `academic_chat_api.py` em vez de respostas simuladas. O servidor executa um fluxo RAG: consulta primeiro o [OpenAlex](https://openalex.org/) (diretório acadêmico aberto, com DOI, periódico, autores e instituições), complementa com [Crossref](https://www.crossref.org/) quando necessário e envia somente esse contexto ao modelo local. A resposta mostra links para a fonte original e para o registro OpenAlex.

O modelo não é fixado na aplicação. O padrão é o [Ollama](https://ollama.com/) com `qwen2.5:7b`, mas também funcionam vLLM e LocalAI por serem compatíveis com a API OpenAI:

```bash
ollama pull qwen2.5:7b
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-chat.txt
python academic_chat_api.py
```

Abra `http://localhost:5000`. Para trocar o modelo ou o servidor, copie `.env.example` para `.env` e exporte as variáveis antes de iniciar:

```bash
export OPEN_MODEL_BASE_URL=http://localhost:11434/v1
export OPEN_MODEL=qwen2.5:7b
export OPEN_MODEL_API_KEY=
python academic_chat_api.py
```

Com Docker, o Compose já aponta o container para o Ollama no host via `host.docker.internal`:

```bash
docker compose up --build
```

Se o modelo estiver indisponível, o sistema não inventa uma resposta: entrega um resumo extrativo das fontes encontradas e informa a falha. Se a busca não encontrar trabalhos verificáveis, informa isso e pede uma consulta mais específica. O endpoint de diagnóstico é `GET /api/health`.
