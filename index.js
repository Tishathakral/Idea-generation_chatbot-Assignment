import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from "@google/generative-ai";
import chalk from 'chalk';
import ora from 'ora';
import prompt from 'prompt-sync';
import fetch from 'node-fetch';
import { configDotenv } from "dotenv";

global.fetch = fetch;

const promptSync = prompt();
configDotenv();

const MODEL_NAME = "gemini-1.0-pro";
const API_KEY = process.env.GOOGLE_API_KEY;
const GENERATION_CONFIG = {
    temperature: 0.9,
    topK: 1,
    topP: 1,
    maxOutputTokens: 2048,
};

const SAFETY_SETTINGS = [
    { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE },
];

const MAX_RETRIES = 3;
const INITIAL_RETRY_DELAY = 1000;

class ConversationHistory {
    constructor() {
        this.history = [];
        this.currentIndex = -1;
    }

    addQuestion(question) {
        if (this.currentIndex < this.history.length - 1) {
            this.history = this.history.slice(0, this.currentIndex + 1);
        }
        
        this.history.push({
            question: question,
            ideas: null,
            lastSelection: null,
            timestamp: new Date()
        });
        this.currentIndex = this.history.length - 1;
    }

    goBack() {
        if (this.currentIndex > 0) {
            this.currentIndex--;
            return this.history[this.currentIndex];
        }
        return null;
    }

    getCurrentQuestion() {
        if (this.currentIndex >= 0) {
            return this.history[this.currentIndex];
        }
        return null;
    }

    updateCurrentIdeas(ideas) {
        if (this.currentIndex >= 0) {
            this.history[this.currentIndex].ideas = ideas;
        }
    }

    updateCurrentSelection(selection) {
        if (this.currentIndex >= 0) {
            this.history[this.currentIndex].lastSelection = selection;
        }
    }
}

const conversationHistory = new ConversationHistory();

function containsInappropriateContent(text) {
    const inappropriatePatterns = [
        /\b(sex|porn|nude|explicit|nsfw)\b/i,
        /\b(terrorist|bomb|attack plans)\b/i,
        /\b(illegal drugs|cocaine|heroin)\b/i,
    ];

    return inappropriatePatterns.some(pattern => pattern.test(text));
}

function displayWelcomeMessage() {
    console.log(chalk.cyan('\n================================================='));
    console.log(chalk.cyan('🤖 Welcome to the Idea Generation Assistant! 🚀'));
    console.log(chalk.cyan('=================================================\n'));
    console.log(chalk.white('How to use this assistant:'));
    console.log(chalk.white('1. Ask any question about what to build or create'));
    console.log(chalk.white('2. Get 3 creative ideas as response'));
    console.log(chalk.white('3. Select your favorite idea(s) by number'));
    console.log(chalk.white('4. Receive detailed guidance for your selection\n'));
    console.log(chalk.white('Special commands:'));
    console.log(chalk.gray('- Type "back" to go to previous question'));
    console.log(chalk.gray('- Type "retry" to generate new ideas for current question'));
    console.log(chalk.gray('- Type "exit" to quit the program\n'));
    console.log(chalk.white('Example questions you can ask:'));
    console.log(chalk.gray('- What app should I build?'));
    console.log(chalk.gray('- What business can I start with $5000?'));
    console.log(chalk.gray('- What website should I create for my portfolio?\n'));
    console.log(chalk.yellow('Note: Please keep questions appropriate and professional.\n'));
    console.log(chalk.cyan('=================================================\n'));
    console.log(chalk.yellow('Please ask your question below:'));
}

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function generateIdeas(chat, question, retryCount = 0) {
    const maxAttempts = 3;
    let attempts = 0;

    while (attempts < maxAttempts) {
        try {
            const initialPrompt = `Generate exactly 3 unique, creative, and practical numbered ideas in response to this question: "${question}". 
                                 Focus on professional and constructive suggestions only.
                                 Make sure each idea is different and specific.
                                 Just list the ideas briefly without any additional details.`;
            
            const response = await sendMessageWithRetry(chat, initialPrompt);
            const ideas = response.text().split('\n').filter(line => line.trim());

            if (ideas.length === 3 && 
                ideas.every(idea => idea.trim().length > 0) &&
                !ideas.some(idea => containsInappropriateContent(idea))) {
                return ideas;
            }
            
            console.log(chalk.yellow(`\nRetrying to generate better ideas (Attempt ${attempts + 1}/${maxAttempts})...`));
            attempts++;
        } catch (error) {
            console.error(chalk.red(`\nError generating ideas (Attempt ${attempts + 1}/${maxAttempts})`));
            attempts++;
            await sleep(1000);
        }
    }

    throw new Error("Unable to generate appropriate ideas after multiple attempts. Please try a different question.");
}

async function sendMessageWithRetry(chat, message, retryCount = 0) {
    try {
        const result = await chat.sendMessage(message);
        return result.response;
    } catch (error) {
        if (error.message.includes('429') && retryCount < MAX_RETRIES) {
            const delay = INITIAL_RETRY_DELAY * Math.pow(2, retryCount);
            console.log(chalk.yellow(`Rate limit hit. Waiting ${delay/1000} seconds before retry...`));
            await sleep(delay);
            return sendMessageWithRetry(chat, message, retryCount + 1);
        }
        throw error;
    }
}

function parseNumberList(input) {
    const numbers = input.split(/[,\s]+/)
        .map(num => parseInt(num.trim()))
        .filter(num => !isNaN(num));
    return [...new Set(numbers)].sort((a, b) => a - b);
}

async function getDetailedSuggestions(chat, selectedIdeas, allIdeas) {
    try {
        const selectedIdeasText = selectedIdeas
            .map(index => allIdeas[index - 1])
            .join(', ');
        
        const prompt = `Please provide detailed suggestions and implementation guidelines for the following idea(s): ${selectedIdeasText}. 
                       Focus on professional and constructive guidance.
                       For each idea, include:
                       1. Key features and functionality
                       2. Technical implementation considerations
                       3. Potential challenges and solutions
                       4. Development timeline estimate
                       5. Required resources and technologies`;
        
        return sendMessageWithRetry(chat, prompt);
    } catch (error) {
        throw new Error("Error generating detailed suggestions. Please try selecting different ideas.");
    }
}

async function handleQuestion(chat, question, isRetry = false) {
    try {
        if (containsInappropriateContent(question)) {
            console.log(chalk.red('\n❌ Please keep your questions appropriate and professional. Try asking about business, technology, or creative projects instead.\n'));
            return false;
        }

        let currentQuestion;
        let ideas;

        if (isRetry) {
            currentQuestion = conversationHistory.getCurrentQuestion();
            if (!currentQuestion) {
                console.log(chalk.red('\n❌ No current question found.'));
                return false;
            }
            console.log(chalk.cyan('\nGenerating new ideas for your question... 🤔'));
            ideas = await generateIdeas(chat, currentQuestion.question);
            conversationHistory.updateCurrentIdeas(ideas);
        } else {
            if (question === 'back') {
                currentQuestion = conversationHistory.goBack();
                if (!currentQuestion) {
                    console.log(chalk.yellow('\nNo previous questions available.'));
                    return false;
                }
                console.log(chalk.cyan('\nGoing back to previous question:'));
                console.log(chalk.cyan(`"${currentQuestion.question}"\n`));
                ideas = currentQuestion.ideas;
            } else {
                conversationHistory.addQuestion(question);
                currentQuestion = conversationHistory.getCurrentQuestion();
                console.log(chalk.cyan('\nGenerating ideas for you... 🤔'));
                ideas = await generateIdeas(chat, question);
                conversationHistory.updateCurrentIdeas(ideas);
            }
        }

        console.log(chalk.blue('\n📝 Here are 3 ideas for you:'));
        ideas.forEach(idea => console.log(chalk.blue(idea)));

        while (true) {
            console.log(chalk.yellow('\n👉 Select any number of ideas by entering their numbers:'));
            console.log(chalk.gray('   Examples: "1" or "1 3" or "1,2,3"'));
            console.log(chalk.gray('   Or type "retry" for new ideas, "back" for previous question'));
            
            const selection = promptSync(chalk.green('Your selection: ')).toLowerCase();

            if (selection === 'retry') {
                return handleQuestion(chat, currentQuestion.question, true);
            }

            if (selection === 'back') {
                return handleQuestion(chat, 'back');
            }

            const selectedNumbers = parseNumberList(selection);
            if (selectedNumbers.length === 0 || selectedNumbers.some(num => num < 1 || num > 3)) {
                console.log(chalk.red('\n❌ Please select valid numbers between 1 and 3.'));
                continue;
            }

            conversationHistory.updateCurrentSelection(selectedNumbers);

            console.log(chalk.cyan('\nGenerating detailed suggestions... 🔍'));
            const detailedResponse = await getDetailedSuggestions(chat, selectedNumbers, ideas);
            console.log(chalk.blue('\n📋 Detailed suggestions:'));
            console.log(chalk.blue(detailedResponse.text()));
            return true;
        }
    } catch (error) {
        console.error(chalk.red('\n❌ Error:'), error.message);
        return false;
    }
}

async function runChat() {
    const spinner = ora('Initializing your idea assistant...').start();
    try {
        const genAI = new GoogleGenerativeAI(API_KEY);
        const model = genAI.getGenerativeModel({ model: MODEL_NAME });

        const chat = model.startChat({
            generationConfig: GENERATION_CONFIG,
            safetySettings: SAFETY_SETTINGS,
            history: [],
        });

        spinner.stop();
        displayWelcomeMessage();

        while (true) {
            const userInput = promptSync(chalk.green('You: '));
            
            if (userInput.toLowerCase() === 'exit') {
                console.log(chalk.yellow('\nThank you for using the Idea Generation Assistant. Goodbye! 👋\n'));
                process.exit(0);
            }

            const success = await handleQuestion(chat, userInput);
            if (success) {
                console.log(chalk.yellow('\n✨ You can:'));
                console.log(chalk.yellow('   1. Ask another question'));
                console.log(chalk.yellow('   2. Type "back" to go to previous question'));
                console.log(chalk.yellow('   3. Type "exit" to quit\n'));
            }
        }
    } catch (error) {
        spinner.stop();
        console.error(chalk.red('⚠️ An error occurred:'), error.message);
        process.exit(1);
    }
}

runChat();