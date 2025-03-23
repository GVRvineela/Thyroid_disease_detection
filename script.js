const quizData =[
    {
        question:"Select the option below according to your condition.",
        a: "Weight gain",
        b: "Weight loss",
        c: "Normal condition",
        correct: "c",
    },
    {
        question:"Select the option below according to your condition.",
        a: "abnormally slow heart beat",
        b: "abnormally fast heart beat",
        c: "Normal heart rate",
        correct: "c",
    },
    {
        question:"Select the option below according to your condition.",
        a: "troble tolerating to cold",
        b: "increased sensitive to heat",
        c: "Adopting to any kind of temperature",
        correct: "c",
    },
    {
        question:"Select the option below according to your condition.",
        a: "Dry, flaky skin",
        b: "Warm and moist skin",
        c: "none of the above",
        correct: "c",
    },
    {
        question:"Select the option below according to your condition.",
        a: "increased hunger levels",
        b: "decreased hunger levels",
        c: "normal",
        correct: "c",
    },
    {
        question:"Select the option below according to your condition.",
        a: "excessive sleepiness",
        b: "difficulty to fall asleep",
        c: "normal sleeping pattern",
        correct: "c",
    },
    {
        question:"Select the option below according to your condition.",
        a: "dry hair",
        b: "hairfall",
        c: "normal condition",
        correct: "c",
    },
    {
        question:"Select the option below according to your condition.",
        a: "constipation",
        b: "increased bowel movements",
        c: "regular bowel movements",
        correct: "c",
    },
    {
        question:"Select the option below according to your condition.",
        a: "rapid nail growth",
        b: "brittle nails",
        c: "normal nail growth",
        correct: "c",
    },
    {
        question:"Select the option below according to your condition.",
        a: "Heavy, frequent peroids in women",
        b: "Scanty, infrequent peroids in women",
        c: "normal (female) / none of the above (male)",
        correct: "c",
    },
];

const quiz= document.getElementById('quiz')
const answerEls = document.querySelectorAll('.answer')
const questionE1 = document.getElementById('question')
const a_text = document.getElementById('a_text')
const b_text = document.getElementById('b_text')
const c_text = document.getElementById('c_text')
const submitBtn = document.getElementById('submit')

let currentQuiz = 0
let score = 0

loadQuiz()

function loadQuiz(){

    deselectAnswers()

    const currentQuizData = quizData[currentQuiz]

    questionE1.innerText = currentQuizData.question
    a_text.innerText = currentQuizData.a
    b_text.innerText = currentQuizData.b
    c_text.innerText = currentQuizData.c
}

function deselectAnswers(){
    answerEls.forEach(answerEl => answerEl.checked = false)
}

function getSelected(){
    let answer
    answerEls.forEach(answerE1 => {
        if(answerE1.checked){
            answer = answerE1.id
        }
    })
    return answer
}

submitBtn.addEventListener('click', () => {
    const answer = getSelected()
    if(answer){
        if (answer ===quizData[currentQuiz].correct){
            score++
        }

        currentQuiz++
        if(currentQuiz < quizData.length){
            loadQuiz()
        }
        else{
            if(score>=5){
                quiz.innerHTML = `
            <h2>Possibly normal condition</h2>
            <h3>(To get accurate results sign in to know more.)</h4>

            <button onclick="location.reload()">Reload</button>
            <button onclick="location.signin()">Signin</button>
            `
            }
            else if(score<5){
                quiz.innerHTML = `
            <h2>You may have thyroid</h2>
            <h3>(To get accurate results sign in to know more.)

            <button onclick="location.reload()">Reload</button> <button onclick="location.signin()">Signin</button>
            `
            }
        }
    }
})

const darkModeButton =
document.getElementById('darkModeButton');

darkModeButton.addEventListener('click', () => {

document.body.classList.toggle('dark-mode');
});