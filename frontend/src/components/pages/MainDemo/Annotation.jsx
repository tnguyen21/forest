import { React } from "react";
import styled from "styled-components";

const StyledAnnotation = styled.span`
  font-family: "Roboto", sans-serif;
  color: green;
`;

function Annotation({ text, annotation }) {
  // TODO some logic to style the annotation based on annotation type

  return <StyledAnnotation>{text}</StyledAnnotation>;
}

export default Annotation;
