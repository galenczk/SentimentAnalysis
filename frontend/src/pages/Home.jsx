import { Formik, Field, Form } from "formik";
import axios from "axios";

export default function Home() {
    async function getAnalysis(text) {
        const response = await axios.post(
            "http://127.0.0.1:5000/analyze", text
        );
        if (response.status === 200) {
            const answer = response.data;
            console.log(answer);
        }
    }

    return (
        <>
            <div className={"sticky top-0 z-50"}>
                <header className={"bg-zinc-900 text-white p-2"}>
                    Sentiment Analysis
                </header>
            </div>

            <div className={"flex flex-col h-[92vh] bg-slate-700"}>
                <main className={"flex justify-center mt-24"}>
                    <div id={"input_form"}>
                        <Formik
                            initialValues={{
                                text_input: "",
                            }}
                            onSubmit={async (values) => {
                                getAnalysis(values);
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
                                    <button className={"btn mt-6"} type={"submit"}>
                                        Analyze
                                    </button>
                                </div>
                            </Form>
                        </Formik>
                    </div>
                </main>
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
