import { useState, useRef } from "preact/hooks";
import { Formik, Field, Form } from "formik";
import axios from "axios";

import OutputTerminal from "./outputTerminal";

export default function Home() {
    async function huggingface_analysis(text) {
        if (text.text_input) {
            const response = await axios.post(
                "http://127.0.0.1:5000/huggingface",
                text
            );
            if (response.status === 200) {
                const answer = response.data;
                let sentiment = answer.senti;
                let probability = (answer.prob * 100).toFixed(2);
                printToOutput(`Input: "${text.text_input}"`);
                printToOutput(
                    `BERT: This statement was ${sentiment}, with a ${probability} probability.`
                );
            }
        }
    }

    async function my_model_analysis(text) {
        const response = await axios.post(
            "http://127.0.0.1:5000/mymodel",
            text
        );
        if (response.status === 200) {
            const answer = response.data;
            console.log(answer.senti);
            printToOutput(`Input: "${text.text_input}"`)
            printToOutput(`My Model: This statement was ${answer.senti}`)
        }
    }

    const [activeTab, setActiveTab] = useState(1);
    const [responses, setResponses] = useState([]);

    const outputRef = useRef()
    
    function handleTabClick(tabIndex) {
        setActiveTab(tabIndex);
    }

    function printToOutput(response) {
        setResponses((existingText) => [...existingText, response]);
    }

    return (
        <>
            <div id={"header"} className={"sticky top-0 z-50"}>
                <header className={"bg-zinc-900 text-white p-2"}>
                    Sentiment Analysis - CS 469 - CISZEKG
                </header>
            </div>

            <div
                id={"pipeline"}
                className={"flex flex-col h-[92vh] bg-zinc-300"}>
                <div id={"tabs"} className={"mx-auto flex w-2/3"}>
                    <button
                        className={
                            activeTab === 1
                                ? "btn-huggingface-selected text-left pl-12"
                                : "btn-huggingface"
                        }
                        onClick={() => handleTabClick(1)}>
                        BERT MODEL
                    </button>
                    <button
                        className={
                            activeTab === 2
                                ? "btn-mymodel-selected text-right pr-12"
                                : "btn-mymodel"
                        }
                        onClick={() => handleTabClick(2)}>
                        My Model
                    </button>
                </div>
                <div
                    id={"analysis_form"}
                    className={"mx-auto w-2/3 flex flex-col"}>
                    {activeTab === 1 ? (
                        <>
                            <div
                                id={"hugging_face_analysis"}
                                className={"bg-sky-700 p-8 flex"}>
                                <div className={"px-24 flex-grow"}>
                                    <Formik
                                        initialValues={{
                                            text_input: "",
                                        }}
                                        onSubmit={async (values) => {
                                            huggingface_analysis(values);
                                        }}>
                                        <Form>
                                            <label>
                                                <Field
                                                    type="text"
                                                    as="textarea"
                                                    id="text_input"
                                                    name="text_input"
                                                    className="text-start text-slate-600 resize-none w-full h-24"></Field>
                                            </label>
                                            <div className={"flex"}>
                                                <button
                                                    className={
                                                        "btn-huggingface-analyze mt-6 mr-auto"
                                                    }
                                                    type={"submit"}>
                                                    Analyze
                                                </button>
                                            </div>
                                        </Form>
                                    </Formik>
                                </div>
                            </div>
                        </>
                    ) : (
                        <>
                            <div
                                id={"my_model_analysis"}
                                className={"bg-emerald-700 p-8 flex"}>
                                <div className={"ml-auto px-24 flex-grow"}>
                                    <Formik
                                        initialValues={{
                                            text_input: "",
                                        }}
                                        onSubmit={async (values) => {
                                            my_model_analysis(values);
                                        }}>
                                        <Form>
                                            <label>
                                                <Field
                                                    type="text"
                                                    as="textarea"
                                                    id="text_input"
                                                    name="text_input"
                                                    className="text-start text-slate-600 resize-none w-full h-24"></Field>
                                            </label>
                                            <div className={"flex"}>
                                                <button
                                                    className={
                                                        "btn-mymodel-analyze mt-6 ml-auto"
                                                    }
                                                    type={"submit"}>
                                                    Analyze
                                                </button>
                                            </div>
                                        </Form>
                                    </Formik>
                                </div>
                            </div>
                        </>
                    )}
                </div>

                <div
                    id={"output"}
                    className={"w-2/3 h-full mx-auto flex flex-col"}>
                    <div
                        className={"pt-4 flex flex-col flex-grow max-h-80"}
                        ref={outputRef}>
                        <OutputTerminal messages={responses} />
                    </div>
                    <button className={"mr-auto btn-clear mt-2"} 
                        onClick={() => {
                            setResponses([])
                        }}
                    >
                        Clear Output
                    </button>
                </div>
            </div>

            <div
                className={
                    "flex justify-center bg-zinc-900 p-2 fixed bottom-0 w-full text-white"
                }>
                <footer>Galen Ciszek &copy; 2023</footer>
            </div>
        </>
    );
}
