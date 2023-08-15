import { useState } from "preact/hooks";
import { Formik, Field, Form } from "formik";
import axios from "axios";

export default function Home() {
    async function huggingface_analysis(text) {
        const response = await axios.post(
            "http://127.0.0.1:5000/huggingface",
            text
        );
        if (response.status === 200) {
            const answer = response.data;
            console.log(answer);
        }
    }

    async function my_model_analysis(text) {
        const response = await axios.post(
            "http://127.0.0.1:5000/huggingface",
            text
        );
        if (response.status === 200) {
            const answer = response.data;
            console.log(answer);
        }
    }

    const [activeTab, setActiveTab] = useState(1);

    const handleTabClick = (tabIndex) => {
        setActiveTab(tabIndex);
    };

    return (
        <>
            <div className={"sticky top-0 z-50"}>
                <header className={"bg-zinc-900 text-white p-2"}>
                    Sentiment Analysis
                </header>
            </div>

            <div className={"flex flex-col h-[92vh] bg-slate-700"}>
                <div className={"flex flex-col "}>
                    <div className={"flex"}>
                        <button
                            className={
                                activeTab === 1 ? "text-red-400 flex-grow" : "text-white"
                            }
                            onClick={() => handleTabClick(1)}>
                            HuggingFace Pipeline
                        </button>
                        <button
                            className={
                                activeTab === 2 ? "text-red-400" : "text-white"
                            }
                            onClick={() => handleTabClick(2)}>
                            My Model
                        </button>
                    </div>

                    <main className={"flex flex-col justify-center mt-24"}>
                        <div
                            id={"hugging_face_analysis"}
                            className={"mx-auto"}>
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
                                            className="h-64 w-96 text-start text-slate-600"></Field>
                                    </label>
                                    <div>
                                        <button
                                            className={"btn mt-6"}
                                            type={"submit"}>
                                            Analyze
                                        </button>
                                    </div>
                                </Form>
                            </Formik>
                        </div>
                    </main>
                </div>
            </div>

            <div
                className={
                    "flex justify-center bg-zinc-300 p-0.5 fixed bottom-0 w-full"
                }>
                <footer>Galen Ciszek &copy; 2023</footer>
            </div>
        </>
    );
}
