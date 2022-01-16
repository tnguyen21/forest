import { React, useState } from "react";
import styled from "styled-components";

const StyledAnnotation = styled.p`
  font-family: "Roboto mono", sans-serif;
  color: green;
`;

function Annotation({ text }) {
  const [annotations, setAnotations] = useState([]);

  // TODO some logic should be in place to style annotations

  return <StyledAnnotation>{text}</StyledAnnotation>;
}

export default Annotation;
