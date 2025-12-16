import "./question.scss";
import questionUserIcon from "../../../assets/question_user.svg";
import { stripTranscribeAppendix } from "../../utils/chatText";
const Question = ({ text, label = "You", meta = null }) => {
  const cleanedText = stripTranscribeAppendix(text);
  return (
    <div className="question_wrapper">
      <div className="user_section">
        <img src={questionUserIcon} alt="user icon" />
        <div className="text">{label}</div>
        {meta ? <div className="meta">{meta}</div> : null}
        <div className="line"></div>
      </div>
      <div className="question_text">{cleanedText}</div>
    </div>
  );
};

export default Question;
