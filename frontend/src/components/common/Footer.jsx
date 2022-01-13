import React from "react";
import styled from "styled-components";

import GitHubIcon from "@mui/icons-material/GitHub";
import ArticleIcon from "@mui/icons-material/Article";

const FooterContainer = styled.footer`
  padding: 0 8rem;

  @media screen and (max-width: 600px) {
    & {
      padding: 0;
    }
  }
`;

const Credits = styled.div`
  display: inline-block;
  margin: auto 0.5rem;
  font-size: 0.8rem;
  font-family: "Roboto mono", monospace;

  @media screen and (max-width: 600px) {
    & {
      display: block;
      width: 100%;
    }
  }
`;

const CreditLink = styled.a`
  text-decoration: none;
  color: rgb(50, 87, 209);
  font-weight: bold;

  &:visited {
    color: rgb(50, 87, 209);
  }
`;

const SocialIcon = styled.a`
  float: right;
  color: black;
  margin-right: 10px;

  &:visited {
    color: black;
  }

  @media screen and (max-width: 600px) {
    & {
      float: center;
      margin: auto;
      padding: 0.5rem;
    }
  }
`;

function Footer() {
  return (
    <FooterContainer>
      <Credits>
        Developed by{" "}
        <CreditLink href="http://github.com/tnguyen21">
          Tommy Bui Nguyen
        </CreditLink>{" "}
        for Senior Design at{" "}
        <CreditLink href="https://drexel.edu/cci/">
          Drexel University
        </CreditLink>
      </Credits>
      <SocialIcon href="https://github.com/tnguyen21/trie">
        <GitHubIcon />
      </SocialIcon>
      <SocialIcon href="#">
        <ArticleIcon />
      </SocialIcon>
    </FooterContainer>
  );
}

export default Footer;
