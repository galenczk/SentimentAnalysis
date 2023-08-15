import { useEffect, useRef } from "preact/hooks";

export default function OutputTerminal({ messages }) {
    const outputRef = useRef();

    useEffect(() => {
        outputRef.current.scrollTop = outputRef.current.scrollHeight
    }, [messages])

    return (
        <div className={"bg-zinc-600 flex-grow overflow-y-auto"} ref={outputRef}>
            {messages.length ? (
                messages.map((message, i) => (
                    <div className={"flex text-white"}>
                        <p>></p>
                        <p key={i} className={"mb-2 ml-4"}>
                            {message}
                        </p>
                    </div>
                ))
            ) : (
                <div className={"flex text-white"}>
                    <p>></p>
                    <p className={"ml-4"}> Awaiting input...</p>
                </div>
            )}
        </div>
    );
}
