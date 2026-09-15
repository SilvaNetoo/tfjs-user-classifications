import * as tf from '@tensorflow/tfjs';

async function trainModel(xs, ys) {
    // Criamos um modelo sequencial
    const model = tf.sequential();

    // Primeira camada da rede:
    // entrada de 7 posições (idade normalizada + 3 cores + 3 localizações)

    //80 neuronios = aqui coloquei tudo para isso, por que tenho pouca base de treino
    // quanto mais neuronios mais complexidade a rede pode aprender e consequentemente mais processamento vai usar

    // a ReLUage como um filtro:
    //é como se ela deixasse os dados interessantes seguirem viagem na rede
    //se a informação chegou nesse neurinio é positiva, passa pra frente!
    //se for 0 ou negativa, pode jogar fora, não vai servir para nada
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));

    // Saída: 3 neuronios 
    // um para cada categoria (premium, medium e basic)

    //activation: softmax normaliza a saída em probabilidade
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));


    // Compilador moderno: adam é um treinador pessoal para redes neurais e ajusta os pesos de forma eficiente
    // para aprender com o histórico de erros e acertos.
    // loss: 'categoricalCrossentropy'
    // Ele compara o que o modelo "acha" (os scores de cada vategoria) com a resposta certa
    // a categoria premium vai ser sempre  [1, 0, 0]

    // metrics: ['accuracy'] é uma métrica de avaliação do modelo, que mede a proporção de acertos
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    //Treinamento do modelo
    await model.fit(xs, ys, {
        verbose: 0, // Desabilita o log interno (e usasó callbacks)
        epochs: 100, // Quantidade de vezrs que vai rodar o nosso dataset(lista de pessoas) para treinar a rede neural
        shuffle: true, // embaralha os dados a cada epoch para evitar overfitting, pra não ficar viciado
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                // console.log(
                //     `Epoch: ${epoch}: loss = ${logs.loss}`
                // )
            }
        }
    });

    return model;
}

// ✅ Opção 1: Envolvendo o objeto em parênteses (Retorno implícito)
async function predict(model, pessoa) {
    const tfInput = tf.tensor2d(pessoa);
    const pred = model.predict(tfInput);
    const predArray = await pred.array();
    
    return predArray[0].map((prob, index) => ({ prob, index }));
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Paraíba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Paraíba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)


// Quanto mais dados melhor!
// Assim o algoritmo vai entendender os padrões complexos dos dados
const model = await trainModel(inputXs, outputYs);

const pessoa = { nome: 'zé', idade: 28, cor: 'verde', localizacao: 'Paraíba' }

//Nomalizando a idade da nova pessoa usando o mesmo padrão do treino
// idade_min = 25 , idade_max = 40, então (28-25)/(40-25) = 0.2)

const pessoaTensorNormalizado = [
    [
        0.2, // idade normalizada
        0, // azul
        0, // vermelho
        1, // verde
        0, // São Paulo
        1, // Rio
        0 // Paraíba
    ]
]; // idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Paraíba

const predictions = await predict(model, pessoaTensorNormalizado);
const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]}: ${(p.prob * 100).toFixed(2)}%`)
    .join('\n');

console.log(results)